import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from mujoco import mjx
from mujoco_playground._src import mjx_env

from . import commands, constants, contact, curricula, delay, model, plant, randomize, recovery, rewards, sense
from .env import Velocity

PRESERVE = "AutoResetWrapper_preserve_info"

CROUCH_ANCHOR = {
    "left_hip_pitch": -1.15,
    "left_knee": 1.25,
    "left_ankle": 1.05,
    "right_hip_pitch": 1.15,
    "right_knee": -1.25,
    "right_ankle": -1.05,
}


@struct.dataclass
class RecoveryWeights:
    upright_progress: float = 5.0
    height_progress: float = 30.0
    joint_torque_rate_l2: float = -2e-3


@struct.dataclass
class RecoveryTuning:
    fallen_tilt_deg: float = 40.0
    term_gate_z: float = 0.08
    recovered_tilt_deg: float = 25.0
    recovered_z: float = 0.09
    min_fallen_s: float = 0.5
    fallen_timeout_s: float = 8.0
    com_upward_max_height: float = 0.125
    height_progress_ceiling: float = 0.115
    head_gate_height_low: float = 0.09
    head_gate_height_high: float = 0.11
    head_gate_tilt_full_deg: float = 20.0


@struct.dataclass
class Spawn:
    prone_prob: float = 0.0
    face_down_prob: float = 1.0
    prone_z_min: float = 0.05
    prone_z_max: float = 0.09
    crouch_prob: float = 0.0
    depth_min: float = 0.35
    depth_max: float = 1.0
    pitch_max_deg: float = 55.0
    joint_noise: float = 0.12
    crouch_z_stand: float = 0.115
    crouch_z_deep: float = 0.06


def euler_quat(yaw, pitch, roll):
    cy, sy = jnp.cos(yaw * 0.5), jnp.sin(yaw * 0.5)
    cp, sp = jnp.cos(pitch * 0.5), jnp.sin(pitch * 0.5)
    cr, sr = jnp.cos(roll * 0.5), jnp.sin(roll * 0.5)
    return jnp.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


class VelStand(mjx_env.MjxEnv):
    def __init__(
        self,
        ctrl_dt: float = 0.02,
        sim_dt: float = 0.005,
        episode_length: int = 1000,
        action_scale: float = 1.0,
        weights: rewards.Weights | None = None,
        tuning: rewards.Tuning | None = None,
        ranges: commands.Ranges | None = None,
        spec: randomize.Spec | None = None,
        noise: delay.Noise | None = None,
        envs: int = curricula.UPSTREAM_ENVS,
        impl: str = "jax",
        nconmax: int | None = None,
        njmax: int | None = None,
        stiff: bool = True,
        spawn: Spawn | None = None,
        recovery_weights: RecoveryWeights | None = None,
        recovery_tuning: RecoveryTuning | None = None,
    ):
        self._walk = Velocity(
            variant="allcollisions",
            ctrl_dt=ctrl_dt,
            sim_dt=sim_dt,
            episode_length=episode_length,
            action_scale=action_scale,
            weights=weights,
            tuning=tuning,
            ranges=ranges,
            spec=spec,
            noise=noise,
            envs=envs,
            impl=impl,
            nconmax=nconmax,
            njmax=njmax,
            stiff=stiff,
        )
        self._ctrl_dt = self._walk._ctrl_dt
        self._sim_dt = self._walk._sim_dt
        self._episode_length = self._walk._episode_length
        self._spawn = spawn or Spawn()
        self._recovery_weights = recovery_weights or RecoveryWeights()
        self._recovery_tuning = recovery_tuning or RecoveryTuning()

        names = self._walk._layout.actuators
        crouch_names = list(CROUCH_ANCHOR)
        self._crouch_slots = np.array([names.index(n) for n in crouch_names])
        crouch_anchor = jnp.asarray([CROUCH_ANCHOR[n] for n in crouch_names])
        nu = self._walk._mj_model.nu
        self._crouch_mask = jnp.zeros(nu).at[self._crouch_slots].set(1.0)
        self._crouch_anchor_full = jnp.zeros(nu).at[self._crouch_slots].set(crouch_anchor)

    @property
    def xml_path(self) -> str:
        return str(constants.XMLS / model.SCENES["allcollisions"])

    @property
    def action_size(self) -> int:
        return self._walk.action_size

    @property
    def mj_model(self):
        return self._walk.mj_model

    @property
    def mjx_model(self):
        return self._walk.mjx_model

    def _names(self):
        return list(vars(self._walk._weights)) + [
            "upright_progress", "height_progress", "joint_torque_rate_l2",
            "com_upward_velocity", "fallen_tax", "recovery_success",
        ]

    def reset(self, rng: jax.Array) -> mjx_env.State:
        walk = self._walk
        spawn = self._spawn
        rng, command_key, draw_key, servo_key, pose_key, imu_key, vel_key, mode_key, prone_key, crouch_key = (
            jax.random.split(rng, 10)
        )

        qpos = walk._stand
        quat_upright, height_upright = qpos[3:7], qpos[2]
        if walk._spec is not None:
            quat_upright, height_upright = randomize.tilted(
                pose_key, walk._spec, walk._spec.base_height
            )

        yaw_key, face_key, prone_z_key = jax.random.split(prone_key, 3)
        yaw = jax.random.uniform(yaw_key, minval=-jnp.pi, maxval=jnp.pi)
        cy, sy = jnp.cos(yaw * 0.5), jnp.sin(yaw * 0.5)
        s = 2.0 ** -0.5
        face_down = jnp.array([s * cy, -s * sy, s * cy, s * sy])
        face_up = jnp.array([s * cy, s * sy, -s * cy, s * sy])
        quat_prone = jnp.where(
            jax.random.uniform(face_key) < spawn.face_down_prob, face_down, face_up
        )
        height_prone = jax.random.uniform(
            prone_z_key, minval=spawn.prone_z_min, maxval=spawn.prone_z_max
        )

        lam_key, pitch_key, roll_key, yaw2_key, noise_key, z_key = jax.random.split(crouch_key, 6)
        lam = jax.random.uniform(lam_key, minval=spawn.depth_min, maxval=spawn.depth_max)
        pitch = lam * jnp.radians(spawn.pitch_max_deg) + jax.random.uniform(
            pitch_key, minval=-jnp.radians(10.0), maxval=jnp.radians(10.0)
        )
        pitch = jnp.maximum(pitch, jnp.radians(5.0))
        roll = jax.random.uniform(roll_key, minval=-jnp.radians(8.0), maxval=jnp.radians(8.0))
        yaw2 = jax.random.uniform(yaw2_key, minval=-jnp.pi, maxval=jnp.pi)
        quat_crouch = euler_quat(yaw2, pitch, roll)
        height_crouch = (
            spawn.crouch_z_stand + lam * (spawn.crouch_z_deep - spawn.crouch_z_stand)
            + jax.random.uniform(z_key, minval=0.0, maxval=0.01)
        )
        joint_noise = (jax.random.uniform(noise_key, (walk._mj_model.nu,)) * 2.0 - 1.0) * spawn.joint_noise
        joints_crouch = (
            walk._home + self._crouch_mask * lam * (self._crouch_anchor_full - walk._home) + joint_noise
        )

        u = jax.random.uniform(mode_key)
        prone = u < spawn.prone_prob
        crouch = (u >= spawn.prone_prob) & (u < spawn.prone_prob + spawn.crouch_prob)

        quat = jnp.where(prone, quat_prone, jnp.where(crouch, quat_crouch, quat_upright))
        height = jnp.where(prone, height_prone, jnp.where(crouch, height_crouch, height_upright))
        joints = jnp.where(crouch, joints_crouch, qpos[walk._wiring.qpos_adr])

        qpos = qpos.at[walk._wiring.qpos_adr].set(joints).at[2].set(height).at[3:7].set(quat)

        data = walk._template.replace(qpos=qpos, qvel=jnp.zeros(walk._mj_model.nv))
        data = mjx.forward(walk._mjx_model, data)

        if walk._spec is None:
            servos = plant.rest(walk._mj_model.nu)
            drawn = None
        else:
            servos = randomize.servos(servo_key, walk._spec, walk._mj_model.nu)
            drawn = randomize.draw(draw_key, walk._mjx_model, walk._spec, walk._slots, walk._ctrl_dt)

        zeros = jnp.zeros(walk._mj_model.nu)
        cos_tilt = sense.cos_tilt(data, walk._sensors)
        ceiling = self._recovery_tuning.height_progress_ceiling
        info = {
            "rng": rng,
            PRESERVE: jnp.zeros((), jnp.int32),
            "servos": servos,
            "gait": contact.rest(len(walk._feet)),
            "commands": commands.rest(command_key, walk._ranges, walk._ctrl_dt),
            "last_action": zeros,
            "previous_action": zeros,
            "head_bias": jnp.zeros(len(constants.HEAD_JOINTS)),
            "targets": jnp.tile(walk._home, (delay.ACTION.max_lag + 1, 1)),
            "draw": drawn,
            "actuator_force": zeros,
            "recovery": recovery.rest(cos_tilt, jnp.minimum(data.qpos[2], ceiling)),
        }

        rates, gravity, _, speeds = walk._measure(data, info, None)
        info["joint_vel"] = delay.rest(walk._joint_vel_line, speeds)
        if walk._spec is not None:
            info["imu"] = delay.start(delay.IMU, rates, imu_key)
            info["gravity"] = delay.start(delay.GRAVITY, gravity, vel_key)

        metrics = {f"reward/{name}": jnp.zeros(()) for name in self._names()}
        obs = walk._observe(data, info)
        return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        walk = self._walk
        tune = self._recovery_tuning
        info = dict(state.info)
        info["rng"], lag_key, push_key, command_key, noise_key, sense_key = jax.random.split(
            info["rng"], 6
        )

        elapsed = info[PRESERVE]
        experience = elapsed * walk._envs
        lag = jax.random.randint(
            lag_key, (), walk._substep_lag.min_lag, walk._substep_lag.max_lag + 1
        )

        target = walk._home + action * walk._action_scale
        if info["draw"] is not None:
            target = target - info["draw"].encoder_bias

        data = state.data
        if info["draw"] is not None:
            data, info["draw"] = walk._shove(data, info["draw"], push_key)

        data, servos, history = plant.advance(
            walk._plant_model(info, experience), data, info["servos"], walk._bam,
            target, walk._wiring, walk.n_substeps,
            history=info["targets"], lag=lag,
        )
        info["servos"] = servos
        info["targets"] = history

        touching = contact.touching(data, walk._feet)
        heights = data.site_xpos[walk._foot_sites][:, 2]
        peak = info["gait"].swing_peak
        gait, filtered, landed, air = contact.tally(
            info["gait"], touching, heights, walk._ctrl_dt
        )

        cos_tilt = sense.cos_tilt(data, walk._sensors)
        z = data.qpos[2]
        fallen = cos_tilt < jnp.cos(jnp.radians(tune.fallen_tilt_deg))
        down = (z < tune.term_gate_z) | fallen
        recovered = (
            (cos_tilt > jnp.cos(jnp.radians(tune.recovered_tilt_deg))) & (z > tune.recovered_z)
        )
        height_capped = jnp.minimum(z, tune.height_progress_ceiling)
        updated, upright_progress, height_progress, taxed, success, down_s = recovery.tally(
            info["recovery"], cos_tilt, height_capped, walk._ctrl_dt,
            fallen, down, recovered, tune.min_fallen_s,
        )
        info["recovery"] = updated

        head = data.qpos[walk._wiring.qpos_adr][walk._head_slots]
        head_error = head - (walk._home[walk._head_slots] + info["commands"].head)
        gate = rewards.upright_gate(
            z, cos_tilt, tune.head_gate_height_low, tune.head_gate_height_high,
            tune.head_gate_tilt_full_deg, tune.fallen_tilt_deg,
        )
        info["head_bias"] = rewards.blend(
            info["head_bias"], head_error * gate, walk._ctrl_dt, walk._tuning.head_tau
        )

        force = data.actuator_force
        terms = walk._rewards(data, info, action, touching, air, landed, peak, heights)
        terms["head_pose_bias"] = rewards.head_pose_bias(info["head_bias"]) * gate
        terms["air_time"] = terms["air_time"] * (~fallen * 1.0)
        terms["upright_progress"] = upright_progress
        terms["height_progress"] = height_progress
        terms["joint_torque_rate_l2"] = rewards.action_rate(force, info["actuator_force"])
        terms["com_upward_velocity"] = (
            rewards.com_upward_velocity(data.qvel[2], z, tune.com_upward_max_height) * fallen
        )
        terms["fallen_tax"] = taxed * 1.0
        terms["recovery_success"] = success * 1.0
        info["actuator_force"] = force

        weights = dict(vars(walk._weights))
        weights["head_pose_bias"] = curricula.staircase(experience, curricula.HEAD_POSE_BIAS_WEIGHT)
        weights["action_rate_l2"] = curricula.staircase(experience, curricula.ACTION_RATE_WEIGHT)
        weights["upright_progress"] = self._recovery_weights.upright_progress
        weights["height_progress"] = self._recovery_weights.height_progress
        weights["joint_torque_rate_l2"] = self._recovery_weights.joint_torque_rate_l2
        weights["com_upward_velocity"] = curricula.staircase(experience, curricula.COM_UPWARD_WEIGHT)
        weights["fallen_tax"] = curricula.staircase(experience, curricula.FALLEN_TAX_WEIGHT)
        weights["recovery_success"] = curricula.staircase(experience, curricula.RECOVERY_SUCCESS_WEIGHT)

        total = sum(jnp.nan_to_num(terms[name]) * weights[name] for name in terms)
        reward = total * walk._ctrl_dt

        info["gait"] = gait
        info["previous_action"] = info["last_action"]
        info["last_action"] = action
        info[PRESERVE] = elapsed + 1
        info["commands"] = commands.refresh(
            info["commands"], command_key, walk._ranges, walk._ctrl_dt,
            head_ranges=curricula.head_ranges(experience),
            standing_fraction=curricula.staircase(experience, curricula.STANDING_FRACTION),
        )

        limit = curricula.staircase(experience, curricula.FELL_OVER_LIMIT)
        fell_over = cos_tilt < jnp.cos(limit)
        fallen_too_long = down_s >= tune.fallen_timeout_s
        broken = ~jnp.isfinite(data.qpos).all() | ~jnp.isfinite(data.qvel).all()
        done = (fell_over | fallen_too_long | broken) * 1.0

        metrics = dict(state.metrics)
        for name, value in terms.items():
            metrics[f"reward/{name}"] = value

        obs = walk._observe(data, info, noise_key, sense_key, elapsed)
        return mjx_env.State(data, obs, reward, done, metrics, info)
