import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from mujoco import mjx
from mujoco_playground._src import mjx_env

from . import constants, contact, curricula, delay, model, plant, randomize, rewards, sense
from . import terrain as rough
from .env import PRESERVE
from .rollerdrive import Drive

FLAT_LENGTH = 1.5
RAMP_LENGTH = 3.0
RUNOUT_LENGTH = 3.0
TOTAL_LENGTH = FLAT_LENGTH + RAMP_LENGTH + RUNOUT_LENGTH

DEG_MIN = 2.0
DEG_MAX = 10.0
ROWS = 8
RESOLUTION = 0.08

SPAWN_ON_RAMP = 0.3
ENTRY_VELOCITY_X = (0.25, 0.45)
WHEEL_RADIUS = 0.0175

FALL_LIMIT = 1.0
VOID_MARGIN = 0.5

PROMOTE_FRACTION = 0.4
DEMOTE_FRACTION = 0.2


def _shape(x: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.clip(x - FLAT_LENGTH, 0.0, None) / RAMP_LENGTH, 0.0, 1.0)


def build_slope_terrain() -> rough.Terrain:
    radius = TOTAL_LENGTH
    grid = int(round(2 * radius / RESOLUTION)) + 1
    axis = np.linspace(-radius, radius, grid)
    row = _shape(axis)
    pattern = np.tile(row[None, :], (grid, 1)).astype(np.float64)

    levels = np.arange(ROWS)
    degrees = DEG_MIN + levels / max(ROWS - 1, 1) * (DEG_MAX - DEG_MIN)
    amplitudes = RAMP_LENGTH * np.tan(np.radians(degrees))
    variants = np.stack([(pattern * a).reshape(-1) for a in amplitudes])

    config = rough.Config(
        rows=ROWS, radius=radius, resolution=RESOLUTION,
        max_height=float(amplitudes[-1]), base_thickness=0.05,
    )
    return rough.Terrain(
        config=config,
        grid=grid,
        pattern=jnp.asarray(pattern),
        variants=jnp.asarray(variants),
        amplitudes=jnp.asarray(amplitudes),
        promote_radius=PROMOTE_FRACTION * TOTAL_LENGTH,
        bounds_radius=TOTAL_LENGTH,
    )


@struct.dataclass
class Weights:
    upright: float = 3.0
    alive: float = 1.0
    wheel_glide: float = 2.0
    heading_hold: float = 1.5
    feet_flat: float = -2.0
    neck_action_rate_l2: float = -0.5
    neck_joint_pos_l2: float = -0.75
    joint_torques_l2: float = -1e-3
    action_rate_l2: float = -1.0


@struct.dataclass
class Tuning:
    upright_std: float = 0.2
    heading_hold_std: float = 0.4
    wheel_glide_cap: float = 0.35
    wheel_radius: float = WHEEL_RADIUS


class RollerSlope(mjx_env.MjxEnv):
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

        self._terrain = build_slope_terrain()
        self._rig = Drive(
            variant="rollers", ctrl_dt=ctrl_dt, sim_dt=sim_dt, envs=envs,
            impl=impl, nconmax=nconmax, njmax=njmax, stiff=stiff,
            terrain=self._terrain,
        )
        self._hfield_variants = self._terrain.variants

        self._spawn_x = FLAT_LENGTH + SPAWN_ON_RAMP
        spawn_shape = float(_shape(np.asarray([self._spawn_x]))[0])
        self._spawn_height = spawn_shape * float(self._terrain.amplitudes[0])
        self._expected_distance = DEMOTE_FRACTION * TOTAL_LENGTH

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
        rng, draw_key, servo_key, entry_key, imu_key, vel_key = jax.random.split(rng, 6)

        qpos = self._rig.stand.at[0].set(self._spawn_x).at[2].add(self._spawn_height)

        entry_speed = jax.random.uniform(
            entry_key, minval=ENTRY_VELOCITY_X[0], maxval=ENTRY_VELOCITY_X[1]
        )
        qvel = jnp.zeros(self._rig.mj_model.nv).at[0].set(entry_speed)
        qvel = qvel.at[jnp.asarray(self._rig.wheels)].set(entry_speed / WHEEL_RADIUS)

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
            PRESERVE: rough.Progress(experience=jnp.zeros((), jnp.int32), level=jnp.zeros((), jnp.int32)),
            "servos": servos,
            "gait": contact.rest(len(self._rig.feet)),
            "heading_ref": sense.yaw(sense.root_quat(data)),
            "last_action": zeros,
            "previous_action": zeros,
            "targets": jnp.tile(self._rig.home, (delay.ACTION.max_lag + 1, 1)),
            "draw": drawn,
            "promoted": jnp.zeros((), dtype=bool),
        }

        rates, gravity, _, speeds = self._measure(data, info, None)
        info["joint_vel"] = delay.rest(self._rig.joint_vel_line, speeds)
        if self._spec is not None:
            info["imu"] = delay.start(delay.IMU, rates, imu_key)
            info["gravity"] = delay.start(delay.GRAVITY, gravity, vel_key)

        metrics = {f"reward/{name}": jnp.zeros(()) for name in vars(self._weights)}
        obs = self._observe(data, info)
        return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def _plant_model(self, info):
        level = info[PRESERVE].level
        swapped = self._rig.mjx_model.tree_replace({"hfield_data": self._hfield_variants[level]})
        if info["draw"] is None:
            return swapped
        return randomize.apply(
            swapped, info["draw"], self._rig.slots,
            curricula.staircase(info[PRESERVE].experience * self._envs, curricula.ROLLER_COM_RANGE),
            curricula.staircase(info[PRESERVE].experience * self._envs, curricula.ROLLER_COM_RANGE),
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        info["rng"], lag_key, push_key, noise_key, sense_key = jax.random.split(info["rng"], 5)

        elapsed = info[PRESERVE].experience
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
            self._plant_model(info), data, info["servos"], self._rig.bam,
            target, self._rig.wiring, self.n_substeps,
            history=info["targets"], lag=lag,
        )
        info["servos"] = servos
        info["targets"] = history

        touching = contact.touching(data, self._rig.feet)
        heights = data.site_xpos[self._rig.foot_sites][:, 2]
        gait, filtered, landed, air = contact.tally(info["gait"], touching, heights, self._ctrl_dt)
        info["gait"] = gait

        terms = self._rewards(data, info, action, touching)
        weights = dict(vars(self._weights))
        total = sum(jnp.nan_to_num(terms[name]) * weights[name] for name in terms)
        reward = total * self._ctrl_dt

        info["previous_action"] = info["last_action"]
        info["last_action"] = action

        fell = sense.tilt(data, self._rig.sensors) > np.sin(FALL_LIMIT) ** 2
        void = data.qpos[2] < -VOID_MARGIN
        broken = ~jnp.isfinite(data.qpos).all() | ~jnp.isfinite(data.qvel).all()
        done = (fell | void | broken) * 1.0

        distance = jnp.clip(data.qpos[0], min=0.0)
        level, promoted, _, _ = rough.advance(
            info[PRESERVE].level, info["promoted"], distance,
            self._terrain.promote_radius, done > 0, self._expected_distance,
            ROWS,
        )
        info["promoted"] = promoted
        info[PRESERVE] = info[PRESERVE].replace(experience=elapsed + 1, level=level)

        metrics = dict(state.metrics)
        for name, value in terms.items():
            metrics[f"reward/{name}"] = value

        obs = self._observe(data, info, noise_key, sense_key, elapsed)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _rewards(self, data, info, action, touching):
        spans = self._rig.sensors
        tune = self._tuning
        joints = data.qpos[self._rig.wiring.qpos_adr]
        yaw = sense.yaw(sense.root_quat(data))
        xmat = data.site_xmat[self._rig.foot_sites]
        torque = data.qfrc_actuator[self._rig.wiring.qvel_adr]

        omega_mean = jnp.mean(data.qvel[jnp.asarray(self._rig.wheels)])

        return {
            "upright": rewards.upright(sense.tilt(data, spans), tune.upright_std),
            "alive": jnp.ones(()),
            "wheel_glide": rewards.wheel_glide(omega_mean, tune.wheel_radius, tune.wheel_glide_cap),
            "heading_hold": rewards.heading_hold(yaw, info["heading_ref"], tune.heading_hold_std),
            "feet_flat": rewards.feet_flat(xmat, sense.GRAVITY, touching),
            "neck_action_rate_l2": rewards.action_rate(
                action[self._rig.head_slots], info["last_action"][self._rig.head_slots]
            ),
            "neck_joint_pos_l2": rewards.joint_deviation(
                joints[self._rig.head_slots], self._rig.home[self._rig.head_slots]
            ),
            "joint_torques_l2": rewards.joint_torques(torque),
            "action_rate_l2": rewards.action_rate(action, info["last_action"]),
        }

    def _measure(self, data, info, key):
        return self._rig.measure(data, info, key, self._noise)

    def _lag(self, info, rates, gravity, speeds, key, elapsed):
        return self._rig.lag(info, rates, gravity, speeds, key, elapsed)

    def _observe(self, data, info, key=None, lag_key=None, elapsed=0):
        spans = self._rig.sensors
        rates, gravity, measured, speeds = self._measure(data, info, key)
        info, rates, gravity, speeds = self._lag(info, rates, gravity, speeds, lag_key, elapsed)

        zero_command = jnp.zeros(3 + constants.HEAD_COMMAND + constants.BODY_COMMAND)

        state = jnp.concatenate([
            rates,
            gravity,
            measured - self._rig.home,
            speeds,
            info["last_action"],
            zero_command,
        ])

        truth = jnp.concatenate([
            sense.read(data, spans.ang_vel),
            sense.gravity(data, spans),
            data.qpos[self._rig.wiring.qpos_adr] - self._rig.home,
            data.qvel[self._rig.wiring.qvel_adr],
            info["last_action"],
            zero_command,
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
