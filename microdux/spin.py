import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct
from mujoco import mjx
from mujoco_playground._src import mjx_env

from . import actuator, constants, contact, curricula, delay, model, plant, randomize, rewards, sense
from .env import NCONMAX, NJMAX, PRESERVE, TILT_LIMIT

SPIN_PERIOD = 4.0
SPIN_RATE_MAX = 3.0
SPIN_ACCEL_END = 0.125
SPIN_HOLD_END = 0.525
SPIN_BRAKE_END = 0.650
SPIN_LAUNCH_DRIFT_SCALE = 0.2
SPIN_WHEEL_OMEGA_SCALE = 17.0

NECK_NO_YAW = ("neck_pitch", "head_pitch", "head_roll")
ANTISYM_BASES = ("hip_pitch", "knee")


@struct.dataclass
class Weights:
    upright: float = 2.0
    body_ang_vel: float = -0.05
    action_rate_l2: float = -1.0
    spin_rate_track: float = 6.0
    spin_rate_l1: float = 0.5
    spin_stay_in_place: float = -3.0
    spin_wheel_differential: float = 1.0
    leg_antisymmetry: float = 1.0
    spin_grounded: float = 0.5
    feet_flat: float = -2.0
    self_collisions: float = -1.0
    neck_action_rate_l2: float = -0.5
    neck_joint_pos_l2: float = -0.2
    joint_torques_l2: float = -1e-3


@struct.dataclass
class Tuning:
    upright_std: float = 0.2**0.5
    spin_rate_std: float = 1.5


def dof(mj_model, name: str) -> int:
    joint = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(mj_model.jnt_dofadr[joint])


class Spin(mjx_env.MjxEnv):
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
        self._nconmax = nconmax if nconmax is not None else NCONMAX
        self._njmax = njmax if njmax is not None else NJMAX
        self._ctrl_dt = ctrl_dt
        self._sim_dt = sim_dt
        self._episode_length = episode_length
        self._action_scale = action_scale
        self._weights = weights or Weights()
        self._tuning = tuning or Tuning()
        self._spec = spec
        self._noise = noise

        self._impl = impl
        self._mj_model, self._layout = model.build(
            "rollers", timestep=sim_dt, iterations=10, ls_iterations=20, stiff=stiff
        )
        self._mjx_model = model.to_mjx(self._mj_model, impl)
        self._template = mjx_env.make_data(
            self._mj_model, impl=impl,
            naconmax=self._nconmax * envs, njmax=self._njmax,
        )
        self._wiring = plant.wire(self._mj_model, self._layout)
        self._sensors = sense.sensors(self._mj_model)
        self._bam = actuator.load(kp=200.0)
        self._slots = randomize.slots(self._mj_model, self._layout)

        self._home = jnp.asarray(self._layout.home)
        self._foot_sites = np.asarray(self._layout.foot_sites)
        self._stand = jnp.asarray(self._mj_model.key_qpos[self._layout.keyframes["STAND"]])
        self._self_collision = tuple(
            int(i) for i in range(self._mj_model.ngeom) if self._mj_model.geom_contype[i] == 2
        )
        self._feet = (
            contact.subtree(self._mj_model, "ankle_l_v1"),
            contact.subtree(self._mj_model, "ankle_r_v1"),
        )

        names = self._layout.actuators
        self._head_slots = np.array([names.index(j) for j in constants.HEAD_JOINTS])
        self._neck_no_yaw_slots = np.array([names.index(j) for j in NECK_NO_YAW])
        self._antisym_left = np.array([names.index(f"left_{base}") for base in ANTISYM_BASES])
        self._antisym_right = np.array([names.index(f"right_{base}") for base in ANTISYM_BASES])

        self._wheel_lf = dof(self._mj_model, "passive_LF_wheel")
        self._wheel_lr = dof(self._mj_model, "passive_LR_wheel")
        self._wheel_rf = dof(self._mj_model, "passive_RF_wheel")
        self._wheel_rr = dof(self._mj_model, "passive_RR_wheel")

        self._substep_lag = delay.ACTION
        self._joint_vel_line = delay.Line(
            min_lag=delay.JOINT_VEL.min_lag, max_lag=delay.JOINT_VEL.max_lag,
            update_period=0, width=self._mj_model.nu,
        )

    @property
    def xml_path(self) -> str:
        return str(constants.XMLS / model.SCENES["rollers"])

    @property
    def action_size(self) -> int:
        return self._mj_model.nu

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, draw_key, servo_key, pose_key, entry_key, imu_key, vel_key = jax.random.split(rng, 7)

        qpos = self._stand.at[2].set(sum(constants.ROLLER_HEIGHT) / 2)
        entry_vx = jnp.zeros(())
        if self._spec is not None:
            quat, height = randomize.tilted(pose_key, self._spec, constants.ROLLER_HEIGHT)
            qpos = qpos.at[2].set(height).at[3:7].set(quat)
            entry_vx = jax.random.uniform(
                entry_key, minval=self._spec.entry_velocity[0], maxval=self._spec.entry_velocity[1]
            )

        qvel = jnp.zeros(self._mj_model.nv).at[0].set(entry_vx)

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
            "phase": jnp.zeros(()),
            "last_action": zeros,
            "previous_action": zeros,
            "targets": jnp.tile(self._home, (delay.ACTION.max_lag + 1, 1)),
            "draw": drawn,
        }

        rates, gravity, _, speeds = self._measure(data, info, None)
        info["joint_vel"] = delay.rest(self._joint_vel_line, speeds)
        if self._spec is not None:
            info["imu"] = delay.start(delay.IMU, rates, imu_key)
            info["gravity"] = delay.start(delay.GRAVITY, gravity, vel_key)

        metrics = {f"reward/{name}": jnp.zeros(()) for name in vars(self._weights)}
        obs = self._observe(data, info)
        return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def _plant_model(self, info, elapsed):
        if info["draw"] is None:
            return self._mjx_model
        return randomize.apply(
            self._mjx_model, info["draw"], self._slots,
            curricula.staircase(elapsed, curricula.ROLLER_COM_RANGE),
            curricula.staircase(elapsed, curricula.ROLLER_COM_RANGE),
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        info["rng"], lag_key, push_key, noise_key, sense_key = jax.random.split(info["rng"], 5)

        elapsed = info[PRESERVE]
        experience = elapsed * self._envs
        lag = jax.random.randint(
            lag_key, (), self._substep_lag.min_lag, self._substep_lag.max_lag + 1
        )

        raw_target = self._home + action * self._action_scale
        target = raw_target
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

        touching = contact.touching(data, self._feet)
        heights = data.site_xpos[self._foot_sites][:, 2]
        peak = info["gait"].swing_peak
        gait, filtered, landed, air = contact.tally(
            info["gait"], touching, heights, self._ctrl_dt
        )

        phase = (info["phase"] + self._ctrl_dt / SPIN_PERIOD) % 1.0

        terms = self._rewards(data, info, action, touching, phase, raw_target)
        weights = dict(vars(self._weights))
        weights["action_rate_l2"] = curricula.staircase(experience, curricula.SPIN_ACTION_RATE_WEIGHT)
        weights["leg_antisymmetry"] = curricula.staircase(experience, curricula.SPIN_LEG_ANTISYM_WEIGHT)
        total = sum(jnp.nan_to_num(terms[name]) * weights[name] for name in terms)
        reward = total * self._ctrl_dt

        info["gait"] = gait
        info["phase"] = phase
        info["previous_action"] = info["last_action"]
        info["last_action"] = action
        info[PRESERVE] = elapsed + 1

        fell = sense.tilt(data, self._sensors) > np.sin(TILT_LIMIT) ** 2
        broken = ~jnp.isfinite(data.qpos).all() | ~jnp.isfinite(data.qvel).all()
        done = (fell | broken) * 1.0

        metrics = dict(state.metrics)
        for name, value in terms.items():
            metrics[f"reward/{name}"] = value

        obs = self._observe(data, info, noise_key, sense_key, elapsed)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _shove(self, data, drawn, key):
        kick_key, timer_key = jax.random.split(key)
        due = drawn.push_timer <= 0
        kick = randomize.shove(kick_key, self._spec)
        qvel = data.qvel.at[:2].add(jnp.where(due, kick, jnp.zeros(2)))
        timer = jnp.where(
            due, randomize.interval(timer_key, self._spec.push_interval, self._ctrl_dt),
            drawn.push_timer - 1,
        )
        return data.replace(qvel=qvel), drawn.replace(push_timer=timer)

    def _rewards(self, data, info, action, touching, phase, raw_target):
        spans = self._sensors
        joints = data.qpos[self._wiring.qpos_adr]
        tune = self._tuning

        omega_z = sense.root_angular_velocity(data)[2]
        v_xy = sense.root_linear_velocity(data)[:2]
        target_rate = rewards.spin_envelope(
            phase, SPIN_RATE_MAX, SPIN_ACCEL_END, SPIN_HOLD_END, SPIN_BRAKE_END
        )
        gate = rewards.spin_gate(
            phase, SPIN_RATE_MAX, SPIN_ACCEL_END, SPIN_HOLD_END, SPIN_BRAKE_END
        )
        launch_scale = jnp.where(phase < SPIN_ACCEL_END, SPIN_LAUNCH_DRIFT_SCALE, 1.0)

        omega_left = (data.qvel[self._wheel_lf] + data.qvel[self._wheel_lr]) / 2.0
        omega_right = (data.qvel[self._wheel_rf] + data.qvel[self._wheel_rr]) / 2.0

        n_contact = jnp.sum(touching * 1.0)
        xmat = data.site_xmat[self._foot_sites]
        torque = data.qfrc_actuator[self._wiring.qvel_adr]

        return {
            "upright": rewards.upright(sense.tilt(data, spans), tune.upright_std),
            "body_ang_vel": rewards.body_angular_velocity(sense.world_angular_velocity(data)),
            "action_rate_l2": rewards.action_rate(action, info["last_action"]),
            "spin_rate_track": rewards.spin_rate_track(omega_z, target_rate, tune.spin_rate_std),
            "spin_rate_l1": rewards.spin_rate_l1(omega_z, target_rate),
            "spin_stay_in_place": rewards.spin_stay_in_place(v_xy, launch_scale),
            "spin_wheel_differential": rewards.spin_wheel_differential(
                omega_right - omega_left, gate, SPIN_WHEEL_OMEGA_SCALE
            ),
            "leg_antisymmetry": rewards.leg_antisymmetry(
                joints[self._antisym_left], joints[self._antisym_right], gate
            ),
            "spin_grounded": rewards.spin_grounded(n_contact, gate),
            "feet_flat": rewards.feet_flat(xmat, sense.GRAVITY, touching),
            "self_collisions": rewards.self_collision(contact.touching(data, self._self_collision)),
            "neck_action_rate_l2": rewards.action_rate(
                action[self._head_slots], info["last_action"][self._head_slots]
            ),
            "neck_joint_pos_l2": rewards.joint_deviation(
                joints[self._neck_no_yaw_slots], self._home[self._neck_no_yaw_slots]
            ),
            "joint_torques_l2": rewards.joint_torques(torque),
        }

    def _measure(self, data, info, key):
        spans = self._sensors
        rates = sense.read(data, spans.ang_vel)
        gravity = sense.gravity(data, spans)
        measured = data.qpos[self._wiring.qpos_adr]
        speeds = data.qvel[self._wiring.qvel_adr]

        if info["draw"] is not None:
            measured = measured + info["draw"].encoder_bias
            rates = sense.rotate(info["draw"].imu_quat, rates)
            gravity = sense.rotate(info["draw"].imu_quat, gravity)

        if key is not None and self._noise is not None:
            noise = self._noise
            rate_key, gravity_key, pos_key, vel_key = jax.random.split(key, 4)
            rates = delay.jitter(rate_key, rates, noise.ang_vel)
            gravity = delay.jitter(gravity_key, gravity, noise.gravity)
            measured = delay.jitter(pos_key, measured, noise.joint_pos)
            speeds = delay.jitter(vel_key, speeds, noise.joint_vel)

        return rates, gravity, measured, speeds

    def _lag(self, info, rates, gravity, speeds, key, elapsed):
        if key is None:
            return info, rates, gravity, speeds

        rate_key, gravity_key, vel_key = jax.random.split(key, 3)
        info["joint_vel"], speeds = delay.push(
            info["joint_vel"], self._joint_vel_line, speeds, vel_key, elapsed
        )
        if "imu" in info:
            info["imu"], rates = delay.push(
                info["imu"], delay.IMU, rates, rate_key, elapsed
            )
            info["gravity"], gravity = delay.push(
                info["gravity"], delay.GRAVITY, gravity, gravity_key, elapsed
            )
        return info, rates, gravity, speeds

    def _observe(self, data, info, key=None, lag_key=None, elapsed=0):
        spans = self._sensors
        rates, gravity, measured, speeds = self._measure(data, info, key)
        info, rates, gravity, speeds = self._lag(
            info, rates, gravity, speeds, lag_key, elapsed
        )

        phase = info["phase"]
        twist = jnp.array([jnp.cos(2 * jnp.pi * phase), jnp.sin(2 * jnp.pi * phase), 0.0])
        command = jnp.concatenate([twist, jnp.zeros(constants.HEAD_COMMAND), jnp.zeros(constants.BODY_COMMAND)])

        state = jnp.concatenate([
            rates,
            gravity,
            measured - self._home,
            speeds,
            info["last_action"],
            command,
        ])

        truth_twist = jnp.array([jnp.cos(2 * jnp.pi * phase), jnp.sin(2 * jnp.pi * phase), 0.0])
        truth = jnp.concatenate([
            sense.read(data, spans.ang_vel),
            sense.gravity(data, spans),
            data.qpos[self._wiring.qpos_adr] - self._home,
            data.qvel[self._wiring.qvel_adr],
            info["last_action"],
            jnp.concatenate([truth_twist, jnp.zeros(constants.HEAD_COMMAND), jnp.zeros(constants.BODY_COMMAND)]),
        ])

        wheel_vel = data.qvel[jnp.array(
            [self._wheel_lf, self._wheel_lr, self._wheel_rf, self._wheel_rr]
        )]
        privileged = jnp.concatenate([
            truth,
            sense.root_linear_velocity(data),
            data.site_xpos[self._foot_sites][:, 2],
            info["gait"].air_time,
            info["gait"].last_contact * 1.0,
            contact.forces(self._mjx_model, data, self._feet).reshape(-1),
            wheel_vel,
        ])
        return {"state": state, "privileged_state": privileged}
