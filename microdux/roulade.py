import math

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from . import commands, constants, contact, curricula, delay, plant, randomize, rewards, sense
from .env import Grounded, PRESERVE

MIDROLL_PITCH_MIN = math.radians(50.0)
MIDROLL_PITCH_MAX = math.radians(340.0)
MIDROLL_OMEGA_RANGE = (0.0, 3.0)
FORWARD_VEL_RANGE = (0.0, 0.0)
TUCK_FACTOR_RANGE = (0.3, 1.0)
JOINT_NOISE_STD = 0.08
STANDING_Z = (0.11, 0.12)
MIDROLL_Z = (0.05, 0.10)
STANDING_TILT_MAX = math.radians(5.0)

HEAD_LATCH_LO = math.radians(20.0)
HEAD_LATCH_HI = math.radians(170.0)
HEAD_TOP_AXIS = jnp.array([0.882, 0.0, 0.471])
HEAD_TOP_DOWN_MIN = 0.3
FLAT_FULL = 0.5
FLAT_ZERO = 0.866
LATERAL_AXIS = jnp.array([0.0, 1.0, 0.0])

TUCK_OVERRIDES = {
    "left_hip_pitch": -1.15,
    "left_knee": 1.25,
    "left_ankle": 1.05,
    "neck_pitch": -1.0,
    "head_pitch": 1.0,
    "right_hip_pitch": 1.15,
    "right_knee": -1.25,
    "right_ankle": -1.05,
}


class Roulade(Grounded):
    def __init__(
        self,
        variant: str = "allcollisions",
        ctrl_dt: float = 0.02,
        sim_dt: float = 0.005,
        episode_length: int = 250,
        action_scale: float = 1.0,
        weights: rewards.RouladeWeights | None = None,
        tuning: rewards.RouladeTuning | None = None,
        ranges: commands.Ranges | None = None,
        spec: randomize.Spec | None = None,
        noise: delay.Noise | None = None,
        envs: int = curricula.UPSTREAM_ENVS,
        impl: str = "jax",
        nconmax: int | None = None,
        njmax: int | None = None,
        stiff: bool = True,
        standing_prob: float = 0.5,
        midroll_prob: float = 0.5,
    ):
        super().__init__(
            variant=variant, ctrl_dt=ctrl_dt, sim_dt=sim_dt, episode_length=episode_length,
            action_scale=action_scale, ranges=ranges, spec=spec, noise=noise, envs=envs,
            impl=impl, nconmax=nconmax, njmax=njmax, stiff=stiff,
        )
        self._weights = weights or rewards.RouladeWeights()
        self._tuning = tuning or rewards.RouladeTuning()
        self._limits = jnp.asarray(
            sense.soft_limits(self._mj_model, self._layout, self._tuning.soft_limit_factor)
        )
        self._standing_prob = standing_prob
        self._midroll_prob = midroll_prob

        self._floor_geom = int(self._layout.floor_geom)
        self._head_body_id = int(
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
        )
        self._head_geoms = contact.subtree(self._mj_model, "jaw_soft")

        names = self._layout.actuators
        tuck_indices = np.array([names.index(n) for n in TUCK_OVERRIDES])
        tuck_targets = np.array([TUCK_OVERRIDES[n] for n in TUCK_OVERRIDES])
        home_np = np.asarray(self._layout.home)
        delta = np.zeros(self._mj_model.nu)
        delta[tuck_indices] = tuck_targets - home_np[tuck_indices]
        self._tuck_delta = jnp.asarray(delta)

    def _spawn(self, key):
        (cat_key, yaw_key, spitch_key, mpitch_key, roll_key, sheight_key,
         mheight_key, tfactor_key, noise_key, omega_key, vel_key) = jax.random.split(key, 11)

        total = jnp.maximum(self._standing_prob + self._midroll_prob, 1e-6)
        is_mid = jax.random.uniform(cat_key) < (self._midroll_prob / total)

        yaw = jax.random.uniform(yaw_key, minval=-jnp.pi, maxval=jnp.pi)
        standing_pitch = jax.random.uniform(
            spitch_key, minval=-STANDING_TILT_MAX, maxval=STANDING_TILT_MAX)
        midroll_pitch = jax.random.uniform(
            mpitch_key, minval=MIDROLL_PITCH_MIN, maxval=MIDROLL_PITCH_MAX)
        pitch = jnp.where(is_mid, midroll_pitch, standing_pitch)
        roll_limit = max(STANDING_TILT_MAX, math.radians(5.0))
        roll = jax.random.uniform(roll_key, minval=-roll_limit, maxval=roll_limit)
        quat = sense.euler_zyx(yaw, pitch, roll)

        standing_z = jax.random.uniform(sheight_key, minval=STANDING_Z[0], maxval=STANDING_Z[1])
        midroll_z = jax.random.uniform(mheight_key, minval=MIDROLL_Z[0], maxval=MIDROLL_Z[1])
        height = jnp.where(is_mid, midroll_z, standing_z)

        u = jax.random.uniform(
            tfactor_key, minval=TUCK_FACTOR_RANGE[0], maxval=TUCK_FACTOR_RANGE[1])
        tuck_joints = self._home + u * self._tuck_delta
        noise = jax.random.uniform(
            noise_key, (self._mj_model.nu,), minval=-JOINT_NOISE_STD, maxval=JOINT_NOISE_STD)
        joints = jnp.where(is_mid, tuck_joints + noise, self._home)

        omega = jax.random.uniform(
            omega_key, minval=MIDROLL_OMEGA_RANGE[0], maxval=MIDROLL_OMEGA_RANGE[1])
        forward = jax.random.uniform(
            vel_key, minval=FORWARD_VEL_RANGE[0], maxval=FORWARD_VEL_RANGE[1])

        qvel = jnp.zeros(self._mj_model.nv)
        qvel = qvel.at[4].set(jnp.where(is_mid, omega, 0.0))
        qvel = qvel.at[0].set(jnp.where(is_mid, 0.0, forward * jnp.cos(yaw)))
        qvel = qvel.at[1].set(jnp.where(is_mid, 0.0, forward * jnp.sin(yaw)))

        spawn_angle = jnp.where(is_mid, midroll_pitch, jnp.zeros(()))
        head_latch = is_mid

        return quat, height, joints, qvel, spawn_angle, head_latch

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, command_key, draw_key, servo_key, spawn_key, imu_key, vel_key = jax.random.split(rng, 7)

        quat, height, joints, qvel, spawn_angle, head_latch = self._spawn(spawn_key)
        qpos = (
            self._stand.at[2].set(height).at[3:7].set(quat).at[self._wiring.qpos_adr].set(joints)
        )

        data = self._template.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self._mjx_model, data)

        if self._spec is None:
            servos = plant.rest(self._mj_model.nu)
            drawn = None
        else:
            servos = randomize.servos(servo_key, self._spec, self._mj_model.nu)
            drawn = randomize.draw(draw_key, self._mjx_model, self._spec, self._slots, self._ctrl_dt)

        zeros = jnp.zeros(self._mj_model.nu)
        info = {
            "rng": rng,
            PRESERVE: jnp.zeros((), jnp.int32),
            "servos": servos,
            "gait": contact.rest(len(self._feet)),
            "commands": commands.rest(command_key, self._ranges, self._ctrl_dt),
            "last_action": zeros,
            "previous_action": zeros,
            "head_bias": jnp.zeros(len(constants.HEAD_JOINTS)),
            "targets": jnp.tile(self._home, (delay.ACTION.max_lag + 1, 1)),
            "draw": drawn,
            "roulade_accum": spawn_angle,
            "roulade_max": spawn_angle,
            "roulade_paid": spawn_angle,
            "roulade_head_latch": head_latch,
            "roulade_torque": zeros,
        }

        rates, gravity, _, speeds = self._measure(data, info, None)
        info["joint_vel"] = delay.rest(self._joint_vel_line, speeds)
        if self._spec is not None:
            info["imu"] = delay.start(delay.IMU, rates, imu_key)
            info["gravity"] = delay.start(delay.GRAVITY, gravity, vel_key)

        metrics = {f"reward/{name}": jnp.zeros(()) for name in vars(self._weights)}
        obs = self._observe(data, info)
        return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def _touch_floor(self, data, geoms):
        geom1, geom2, dist = contact.pairs(data)
        live = dist < 0
        a = contact.member(geom1, geoms)
        b = contact.member(geom2, geoms)
        on_floor1 = geom1 == self._floor_geom
        on_floor2 = geom2 == self._floor_geom
        return jnp.any(live & ((a & on_floor2) | (b & on_floor1)))

    def _accumulate(self, data, info):
        omega_body = sense.root_angular_velocity(data)
        omega_fwd = omega_body[1]
        raw_delta = jnp.nan_to_num(omega_fwd, nan=0.0) * self._ctrl_dt

        supported = contact.touching(data, (self._floor_geom,))[0]
        raw_delta = raw_delta * supported.astype(jnp.float32)

        axis_world = jnp.nan_to_num(
            sense.rotate(sense.root_quat(data), LATERAL_AXIS), nan=1.0
        )
        flat = rewards.smoothstep(
            (FLAT_ZERO - jnp.abs(axis_world[2])) / (FLAT_ZERO - FLAT_FULL)
        )
        raw_delta = raw_delta * flat

        accum = info["roulade_accum"] + raw_delta
        frontier = jnp.maximum(info["roulade_max"], accum)

        head_contact = self._touch_floor(data, self._head_geoms)
        in_head_window = (accum > HEAD_LATCH_LO) & (accum < HEAD_LATCH_HI)
        top_down = sense.rotate(
            data.xquat[self._head_body_id], HEAD_TOP_AXIS
        )[2] < -HEAD_TOP_DOWN_MIN
        head_latch = info["roulade_head_latch"] | (head_contact & in_head_window & top_down)

        target = self._tuning.target_angle
        capped_frontier = jnp.minimum(frontier, target)
        capped_paid = jnp.minimum(info["roulade_paid"], target)
        delta = jnp.clip(
            capped_frontier - capped_paid, 0.0, self._tuning.max_paid_rate * self._ctrl_dt
        )
        paid = jnp.maximum(info["roulade_paid"], capped_frontier)

        info["roulade_accum"] = accum
        info["roulade_max"] = frontier
        info["roulade_paid"] = paid
        info["roulade_head_latch"] = head_latch
        return info, delta, omega_body, axis_world, head_contact, top_down

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        info["rng"], lag_key, push_key, command_key, noise_key, sense_key = jax.random.split(
            info["rng"], 6
        )

        elapsed = info[PRESERVE]
        experience = elapsed * self._envs
        lag = jax.random.randint(
            lag_key, (), self._substep_lag.min_lag, self._substep_lag.max_lag + 1
        )

        target = self._home + action * self._action_scale
        if info["draw"] is not None:
            target = target - info["draw"].encoder_bias

        data = state.data
        if info["draw"] is not None:
            data, info["draw"] = self._shove(data, info["draw"], push_key)

        data, servos, history = plant.advance(
            self._plant_model(info, experience), data, info["servos"], self._bam,
            target, self._wiring, self.n_substeps,
            history=info["targets"], lag=lag,
        )
        info["servos"] = servos
        info["targets"] = history

        head = data.qpos[self._wiring.qpos_adr][self._head_slots]
        head_error = head - (self._home[self._head_slots] + info["commands"].head)
        info["head_bias"] = rewards.blend(
            info["head_bias"], head_error, self._ctrl_dt, self._tuning.head_tau
        )

        info, delta, omega_body, axis_world, head_contact, top_down = self._accumulate(data, info)

        torque = data.qfrc_actuator[self._wiring.qvel_adr]
        terms = self._rewards(
            data, info, action, torque, delta, omega_body, axis_world, head_contact, top_down
        )
        info["roulade_torque"] = torque

        weights = dict(vars(self._weights))
        weights["action_rate_l2"] = curricula.staircase(
            experience, curricula.ROULADE_ACTION_RATE_WEIGHT
        )
        weights["arrival_damping"] = curricula.staircase(
            experience, curricula.ROULADE_ARRIVAL_DAMPING_WEIGHT
        )
        weights["torque_rate_l2"] = curricula.staircase(
            experience, curricula.ROULADE_TORQUE_RATE_WEIGHT
        )
        weights["gentle_landing"] = curricula.staircase(
            experience, curricula.ROULADE_GENTLE_LANDING_WEIGHT
        )
        total = sum(jnp.nan_to_num(terms[name]) * weights[name] for name in terms)
        reward = total * self._ctrl_dt

        info["previous_action"] = info["last_action"]
        info["last_action"] = action
        info[PRESERVE] = elapsed + 1
        info["commands"] = commands.refresh(
            info["commands"], command_key, self._ranges, self._ctrl_dt,
        )

        broken = ~jnp.isfinite(data.qpos).all() | ~jnp.isfinite(data.qvel).all()
        done = broken * 1.0

        metrics = dict(state.metrics)
        for name, value in terms.items():
            metrics[f"reward/{name}"] = value

        obs = self._observe(data, info, noise_key, sense_key, elapsed)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _rewards(self, data, info, action, torque, delta, omega_body, axis_world, head_contact, top_down):
        spans = self._sensors
        tune = self._tuning
        joints = data.qpos[self._wiring.qpos_adr]
        legs = self._leg_slots
        target_legs = self._home[legs]

        height = data.qpos[2]
        vz = data.qvel[2]
        az = data.qacc[2]
        cos_tilt = -sense.gravity(data, spans)[2]
        tilt = sense.tilt(data, spans)

        accum = info["roulade_accum"]
        max_accum = info["roulade_max"]
        head_latch = info["roulade_head_latch"] * 1.0

        landing_gate = (
            rewards.smoothstep(rewards.gate(max_accum, tune.gate_lo, tune.gate_hi)) * head_latch
        )
        rise_gate = (
            rewards.smoothstep(rewards.gate(max_accum, tune.rise_gate_lo, tune.rise_gate_hi))
            * head_latch
        )

        omega_fwd = omega_body[1]
        in_pivot_window = ((accum > tune.head_pivot_lo) & (accum < tune.head_pivot_hi)) * 1.0
        body_ang_vel = rewards.body_angular_velocity(sense.world_angular_velocity(data))

        landing_score = rewards.composite(
            rewards.height_gaussian(height, self._stand[2], tune.landing_height_std),
            rewards.upright(tilt, tune.landing_upright_std),
            rewards.pose_target(joints[legs], target_legs, tune.landing_pose_std),
        )
        sharp_score = rewards.composite(
            rewards.upright(tilt, tune.sharp_upright_std),
            rewards.height_gaussian(height, self._stand[2], tune.sharp_height_std),
        )

        return {
            "progress": rewards.progress_rate(delta, self._ctrl_dt, tune.target_angle),
            "overspeed": rewards.rate_overspeed(omega_fwd, tune.omega_max),
            "head_pivot": rewards.head_pivot_score(
                head_contact * 1.0, in_pivot_window,
                omega_fwd / tune.head_pivot_rate_norm, top_down * 1.0,
            ),
            "landing_composite": landing_score * landing_gate,
            "upright_after_roll": rewards.upright_bootstrap(cos_tilt) * landing_gate,
            "height_after_roll": rewards.height_gaussian(
                height, self._stand[2], tune.height_after_roll_std) * landing_gate,
            "landing_sharp": sharp_score * landing_gate,
            "stand_tax": rewards.height_shortfall(height, self._stand[2]) * landing_gate,
            "rise_velocity": rewards.com_upward_velocity(
                vz, height, self._stand[2] + tune.rise_margin) * rise_gate,
            "sagittal": rewards.off_axis_angular_velocity(omega_body),
            "lateral_vel": rewards.component_sq(sense.root_linear_velocity(data), 1),
            "flatness": rewards.component_sq(axis_world, 2),
            "action_rate_l2": rewards.action_rate(action, info["last_action"]),
            "torque_rate_l2": rewards.action_rate(torque, info["roulade_torque"]),
            "body_ang_vel": body_ang_vel,
            "angular_momentum": rewards.angular_momentum(sense.read(data, spans.angmom)),
            "dof_pos_limits": rewards.joint_pos_limits(joints, self._limits),
            "arrival_damping": body_ang_vel * rewards.upright_gate(
                height, cos_tilt, tune.arrival_height_low, tune.arrival_height_high,
                tune.arrival_tilt_full_deg, tune.arrival_tilt_zero_deg),
            "gentle_landing": rewards.vertical_accel(az),
            "self_collisions": rewards.self_collision(contact.touching(data, self._self_collision)),
            "head_pose_tracking": rewards.head_pose_tracking(
                joints[self._head_slots], info["commands"].head,
                self._home[self._head_slots], tune.head_std),
            "head_pose_bias": rewards.head_pose_bias(info["head_bias"]),
        }
