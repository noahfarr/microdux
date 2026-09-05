import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from mujoco import mjx
from mujoco_playground._src import mjx_env

from . import constants, contact, curricula, delay, model, plant, randomize, rewards, sense
from .env import PRESERVE, TILT_LIMIT
from .rollerdrive import Drive

CROUCH_PERIOD = 5.0
DESCENT_END = 0.10
HOLD_END = 0.50
RISE_END = 0.60
CROUCH_LEAN_PITCH = 0.08

STAND_POSE = {
    "left_hip_yaw": -0.0476, "left_hip_roll": -0.0629, "left_hip_pitch": -0.2869,
    "left_knee": 0.9618, "left_ankle": 1.1674,
    "neck_pitch": 0.6029, "head_pitch": 0.543, "head_yaw": -0.069, "head_roll": -0.0414,
    "right_hip_yaw": -0.0337, "right_hip_roll": -0.0061, "right_hip_pitch": 0.1534,
    "right_knee": -0.9725, "right_ankle": -1.0646,
}

CROUCH_POSE = {
    "left_hip_yaw": -0.0184, "left_hip_roll": 0.0307, "left_hip_pitch": 1.4082,
    "left_knee": 1.5248, "left_ankle": -0.0675,
    "neck_pitch": 1.0937, "head_pitch": 1.2149, "head_yaw": -0.0184, "head_roll": -0.0368,
    "right_hip_yaw": 0.0184, "right_hip_roll": -0.0169, "right_hip_pitch": -1.4757,
    "right_knee": -1.5907, "right_ankle": 0.0568,
}


def pose_vector(pose: dict, names) -> np.ndarray:
    return np.array([pose[n] for n in names])


@struct.dataclass
class Weights:
    upright: float = 2.0
    body_ang_vel: float = -0.05
    angular_momentum: float = -0.02
    action_rate_l2: float = -1.0
    crouch_pose: float = 6.0
    crouch_pose_l1: float = 2.0
    forward_speed: float = 1.0
    crouch_forward_lean: float = 1.0
    feet_flat: float = -2.0
    self_collisions: float = -1.0
    neck_action_rate_l2: float = -0.5
    joint_torques_l2: float = -1e-3


@struct.dataclass
class Tuning:
    upright_std: float = 0.2**0.5
    crouch_pose_std: float = 0.4
    forward_speed_ref: float = 0.2
    crouch_lean_target: float = CROUCH_LEAN_PITCH
    crouch_lean_std: float = 0.1


class RollerCrouch(mjx_env.MjxEnv):
    def __init__(
        self,
        ctrl_dt: float = 0.02,
        sim_dt: float = 0.005,
        episode_length: int = 1000,
        action_scale: float = 1.0,
        weights: Weights | None = None,
        tuning: Tuning | None = None,
        spec: randomize.Spec | None = None,
        noise: delay.Noise | None = None,
        envs: int = curricula.UPSTREAM_ENVS,
        impl: str = "jax",
        nconmax: int | None = None,
        njmax: int | None = None,
        stiff: bool = True,
    ):
        self._envs = envs
        self._ctrl_dt = ctrl_dt
        self._sim_dt = sim_dt
        self._episode_length = episode_length
        self._action_scale = action_scale
        self._weights = weights or Weights()
        self._tuning = tuning or Tuning()
        self._spec = spec
        self._noise = noise

        self._rig = Drive(
            variant="rollers", ctrl_dt=ctrl_dt, sim_dt=sim_dt, envs=envs,
            impl=impl, nconmax=nconmax, njmax=njmax, stiff=stiff,
        )

        names = self._rig.layout.actuators
        self._stand_pose = jnp.asarray(pose_vector(STAND_POSE, names))
        self._crouch_pose = jnp.asarray(pose_vector(CROUCH_POSE, names))

    @property
    def xml_path(self) -> str:
        return str(constants.XMLS / model.SCENES["rollers"])

    @property
    def action_size(self) -> int:
        return self._rig.mj_model.nu

    @property
    def mj_model(self):
        return self._rig.mj_model

    @property
    def mjx_model(self):
        return self._rig.mjx_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, draw_key, servo_key, pose_key, entry_key, imu_key, vel_key = jax.random.split(rng, 7)

        qpos = self._rig.stand.at[2].set(sum(constants.ROLLER_HEIGHT) / 2)
        entry_vx = jnp.zeros(())
        if self._spec is not None:
            quat, height = randomize.tilted(pose_key, self._spec, constants.ROLLER_HEIGHT)
            qpos = qpos.at[2].set(height).at[3:7].set(quat)
            entry_vx = jax.random.uniform(
                entry_key, minval=self._spec.entry_velocity[0], maxval=self._spec.entry_velocity[1]
            )

        qvel = jnp.zeros(self._rig.mj_model.nv).at[0].set(entry_vx)

        data = self._rig.template.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self._rig.mjx_model, data)

        if self._spec is None:
            servos = plant.rest(self._rig.mj_model.nu)
            drawn = None
        else:
            servos = randomize.servos(servo_key, self._spec, self._rig.mj_model.nu)
            drawn = randomize.draw(
                draw_key, self._rig.mjx_model, self._spec, self._rig.slots, self._ctrl_dt
            )

        zeros = jnp.zeros(self._rig.mj_model.nu)
        info = {
            "rng": rng,
            PRESERVE: jnp.zeros((), jnp.int32),
            "servos": servos,
            "gait": contact.rest(len(self._rig.feet)),
            "phase": jnp.zeros(()),
            "last_action": zeros,
            "previous_action": zeros,
            "targets": jnp.tile(self._rig.home, (delay.ACTION.max_lag + 1, 1)),
            "draw": drawn,
        }

        rates, gravity, _, speeds = self._measure(data, info, None)
        info["joint_vel"] = delay.rest(self._rig.joint_vel_line, speeds)
        if self._spec is not None:
            info["imu"] = delay.start(delay.IMU, rates, imu_key)
            info["gravity"] = delay.start(delay.GRAVITY, gravity, vel_key)

        metrics = {f"reward/{name}": jnp.zeros(()) for name in vars(self._weights)}
        obs = self._observe(data, info)
        return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def _plant_model(self, info, elapsed):
        if info["draw"] is None:
            return self._rig.mjx_model
        return self._rig.plant_model(
            info,
            curricula.staircase(elapsed, curricula.ROLLER_COM_RANGE),
            curricula.staircase(elapsed, curricula.ROLLER_COM_RANGE),
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        info["rng"], lag_key, push_key, noise_key, sense_key = jax.random.split(info["rng"], 5)

        elapsed = info[PRESERVE]
        experience = elapsed * self._envs
        lag = jax.random.randint(
            lag_key, (), self._rig.substep_lag.min_lag, self._rig.substep_lag.max_lag + 1
        )

        target = self._rig.home + action * self._action_scale
        if info["draw"] is not None:
            target = target - info["draw"].encoder_bias

        data = state.data
        if info["draw"] is not None:
            data, info["draw"] = self._rig.shove(data, info["draw"], push_key, self._spec)

        data, servos, history = plant.advance(
            self._plant_model(info, experience), data, info["servos"], self._rig.bam,
            target, self._rig.wiring, self.n_substeps,
            history=info["targets"], lag=lag,
        )
        info["servos"] = servos
        info["targets"] = history

        touching = contact.touching(data, self._rig.feet)
        heights = data.site_xpos[self._rig.foot_sites][:, 2]
        gait, filtered, landed, air = contact.tally(
            info["gait"], touching, heights, self._ctrl_dt
        )

        phase = (info["phase"] + self._ctrl_dt / CROUCH_PERIOD) % 1.0

        terms = self._rewards(data, info, action, touching, phase)
        weights = dict(vars(self._weights))
        weights["action_rate_l2"] = curricula.staircase(experience, curricula.SPIN_ACTION_RATE_WEIGHT)
        total = sum(jnp.nan_to_num(terms[name]) * weights[name] for name in terms)
        reward = total * self._ctrl_dt

        info["gait"] = gait
        info["phase"] = phase
        info["previous_action"] = info["last_action"]
        info["last_action"] = action
        info[PRESERVE] = elapsed + 1

        fell = sense.tilt(data, self._rig.sensors) > np.sin(TILT_LIMIT) ** 2
        broken = ~jnp.isfinite(data.qpos).all() | ~jnp.isfinite(data.qvel).all()
        done = (fell | broken) * 1.0

        metrics = dict(state.metrics)
        for name, value in terms.items():
            metrics[f"reward/{name}"] = value

        obs = self._observe(data, info, noise_key, sense_key, elapsed)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _rewards(self, data, info, action, touching, phase):
        spans = self._rig.sensors
        joints = data.qpos[self._rig.wiring.qpos_adr]
        tune = self._tuning

        blend = rewards.crouch_blend(phase, DESCENT_END, HOLD_END, RISE_END)
        target = self._stand_pose + blend * (self._crouch_pose - self._stand_pose)
        lean = sense.gravity(data, spans)[0]
        vx = sense.root_linear_velocity(data)[0]
        xmat = data.site_xmat[self._rig.foot_sites]
        torque = data.qfrc_actuator[self._rig.wiring.qvel_adr]

        return {
            "upright": rewards.upright(sense.tilt(data, spans), tune.upright_std),
            "body_ang_vel": rewards.body_angular_velocity(sense.world_angular_velocity(data)),
            "angular_momentum": rewards.angular_momentum(sense.read(data, spans.angmom)),
            "action_rate_l2": rewards.action_rate(action, info["last_action"]),
            "crouch_pose": rewards.pose_target(joints, target, tune.crouch_pose_std),
            "crouch_pose_l1": rewards.pose_l1(joints, target),
            "forward_speed": rewards.forward_speed(vx, tune.forward_speed_ref),
            "crouch_forward_lean": rewards.forward_lean(
                blend, lean, tune.crouch_lean_target, tune.crouch_lean_std
            ),
            "feet_flat": rewards.feet_flat(xmat, sense.GRAVITY, touching),
            "self_collisions": rewards.self_collision(
                contact.touching(data, self._rig.self_collision)
            ),
            "neck_action_rate_l2": rewards.action_rate(
                action[self._rig.head_slots], info["last_action"][self._rig.head_slots]
            ),
            "joint_torques_l2": rewards.joint_torques(torque),
        }

    def _measure(self, data, info, key):
        return self._rig.measure(data, info, key, self._noise)

    def _lag(self, info, rates, gravity, speeds, key, elapsed):
        return self._rig.lag(info, rates, gravity, speeds, key, elapsed)

    def _observe(self, data, info, key=None, lag_key=None, elapsed=0):
        spans = self._rig.sensors
        rates, gravity, measured, speeds = self._measure(data, info, key)
        info, rates, gravity, speeds = self._lag(info, rates, gravity, speeds, lag_key, elapsed)

        phase = info["phase"]
        twist = jnp.array([jnp.cos(2 * jnp.pi * phase), jnp.sin(2 * jnp.pi * phase), 0.0])
        command = jnp.concatenate([twist, jnp.zeros(constants.HEAD_COMMAND), jnp.zeros(constants.BODY_COMMAND)])

        state = jnp.concatenate([
            rates,
            gravity,
            measured - self._rig.home,
            speeds,
            info["last_action"],
            command,
        ])

        truth = jnp.concatenate([
            sense.read(data, spans.ang_vel),
            sense.gravity(data, spans),
            data.qpos[self._rig.wiring.qpos_adr] - self._rig.home,
            data.qvel[self._rig.wiring.qvel_adr],
            info["last_action"],
            command,
        ])

        wheel_vel = data.qvel[jnp.asarray(self._rig.wheels)]
        privileged = jnp.concatenate([
            truth,
            sense.root_linear_velocity(data),
            data.site_xpos[self._rig.foot_sites][:, 2],
            info["gait"].air_time,
            info["gait"].last_contact * 1.0,
            contact.forces(self._rig.mjx_model, data, self._rig.feet).reshape(-1),
            wheel_vel,
        ])
        return {"state": state, "privileged_state": privileged}
