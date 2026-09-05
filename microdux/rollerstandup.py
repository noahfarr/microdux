import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from . import constants, contact, curricula, delay, model, plant, randomize, rewards, sense
from .env import PRESERVE
from .rollerdrive import Drive

ROLLER_STAND_Z = 0.138
ROLLER_PRONE_Z = 0.075

DEFAULT_GROUND_MIX = randomize.GroundMix(
    standing=0.20, sitting=0.0, face_down=0.40, face_up=0.40,
    standing_height=(0.134, 0.144), prone_height=(0.076, 0.09),
    sitting_tilt=10.0, face_up_roll=0.0,
)


class RollerStandUp(mjx_env.MjxEnv):
    def __init__(
        self,
        ctrl_dt: float = 0.02,
        sim_dt: float = 0.005,
        episode_length: int = 300,
        action_scale: float = 1.0,
        weights: rewards.RollerStandUpWeights | None = None,
        tuning: rewards.RollerStandUpTuning | None = None,
        spec: randomize.Spec | None = None,
        noise: delay.Noise | None = None,
        envs: int = curricula.UPSTREAM_ENVS,
        impl: str = "jax",
        nconmax: int | None = None,
        njmax: int | None = None,
        stiff: bool = True,
        ground_mix: randomize.GroundMix | None = None,
    ):
        self._envs = envs
        self._ctrl_dt = ctrl_dt
        self._sim_dt = sim_dt
        self._episode_length = episode_length
        self._action_scale = action_scale
        self._weights = weights or rewards.RollerStandUpWeights()
        self._tuning = tuning or rewards.RollerStandUpTuning()
        self._spec = spec
        self._noise = noise
        self._ground_mix = ground_mix or DEFAULT_GROUND_MIX

        self._rig = Drive(
            variant="rollers", ctrl_dt=ctrl_dt, sim_dt=sim_dt, envs=envs,
            impl=impl, nconmax=nconmax, njmax=njmax, stiff=stiff,
        )

        names = self._rig.layout.actuators
        self._leg_slots = np.array(
            [i for i, n in enumerate(names) if "head" not in n and "neck" not in n]
        )
        self._limits = jnp.asarray(self._rig.mj_model.jnt_range[self._rig.layout.actuated_joint_ids])

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
        rng, draw_key, servo_key, ground_key, imu_key, vel_key = jax.random.split(rng, 6)

        joints, quat, height = randomize.groundstate(
            ground_key, self._ground_mix, self._rig.home, self._rig.home
        )
        qpos = self._rig.stand.at[2].set(height).at[3:7].set(quat)
        qpos = qpos.at[self._rig.wiring.qpos_adr].set(joints)

        data = self._rig.template.replace(qpos=qpos, qvel=jnp.zeros(self._rig.mj_model.nv))
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
            "last_action": zeros,
            "previous_action": zeros,
            "targets": jnp.tile(self._rig.home, (delay.ACTION.max_lag + 1, 1)),
            "draw": drawn,
            "actuator_force": zeros,
        }

        rates, gravity, _, speeds = self._measure(data, info, None)
        info["joint_vel"] = delay.rest(self._rig.joint_vel_line, speeds)
        if self._spec is not None:
            info["imu"] = delay.start(delay.IMU, rates, imu_key)
            info["gravity"] = delay.start(delay.GRAVITY, gravity, vel_key)

        metrics = {f"reward/{name}": jnp.zeros(()) for name in vars(self._weights)}
        obs = self._observe(data, info)
        return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def _plant_model(self, info, experience):
        friction = curricula.staircase(experience, curricula.ROLLER_STANDUP_WHEEL_FRICTION)
        rolled = self._rig.mjx_model.tree_replace({
            "dof_frictionloss": self._rig.mjx_model.dof_frictionloss.at[self._rig.wheels].set(friction),
        })
        if info["draw"] is None:
            return rolled
        return randomize.apply(
            rolled, info["draw"], self._rig.slots,
            curricula.staircase(experience, curricula.ROLLER_COM_RANGE),
            curricula.staircase(experience, curricula.ROLLER_COM_RANGE),
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
            magnitude = curricula.staircase(experience, curricula.ROLLER_STANDUP_PUSH_MAGNITUDE)
            data, info["draw"] = self._shove(data, info["draw"], push_key, magnitude)

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
        info["gait"] = gait

        force = data.actuator_force
        terms = self._rewards(data, info, action, force)
        weights = dict(vars(self._weights))
        weights["action_rate_l2"] = curricula.staircase(
            experience, curricula.ROLLER_STANDUP_ACTION_RATE_WEIGHT
        )
        total = sum(jnp.nan_to_num(terms[name]) * weights[name] for name in terms)
        reward = total * self._ctrl_dt

        info["previous_action"] = info["last_action"]
        info["last_action"] = action
        info["actuator_force"] = force
        info[PRESERVE] = elapsed + 1

        broken = ~jnp.isfinite(data.qpos).all() | ~jnp.isfinite(data.qvel).all()
        done = broken * 1.0

        metrics = dict(state.metrics)
        for name, value in terms.items():
            metrics[f"reward/{name}"] = value

        obs = self._observe(data, info, noise_key, sense_key, elapsed)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _shove(self, data, drawn, key, magnitude):
        kick_key, timer_key = jax.random.split(key)
        due = drawn.push_timer <= 0
        kick = jax.random.uniform(kick_key, (2,), minval=-magnitude, maxval=magnitude)
        qvel = data.qvel.at[:2].add(jnp.where(due, kick, jnp.zeros(2)))
        timer = jnp.where(
            due, randomize.interval(timer_key, self._spec.push_interval, self._ctrl_dt),
            drawn.push_timer - 1,
        )
        return data.replace(qvel=qvel), drawn.replace(push_timer=timer)

    def _rewards(self, data, info, action, force):
        spans = self._rig.sensors
        tune = self._tuning
        joints = data.qpos[self._rig.wiring.qpos_adr]
        legs = self._leg_slots
        target_legs = self._rig.home[legs]

        height = data.qpos[2]
        vz = data.qvel[2]
        az = data.qacc[2]
        cos_tilt = sense.cos_tilt(data, spans)
        tilt = sense.tilt(data, spans)
        torque = data.qfrc_actuator[self._rig.wiring.qvel_adr]
        raw_target = self._rig.home + action * self._action_scale

        return {
            "pose_legs": rewards.pose_target(joints[legs], target_legs, tune.pose_std),
            "pose_legs_l1": rewards.pose_l1(joints[legs], target_legs),
            "height": rewards.height_gaussian(height, tune.stand_z, tune.height_std),
            "height_sharp": rewards.height_gaussian(height, tune.stand_z, tune.height_sharp_std),
            "height_l1": rewards.height_l1(height, tune.stand_z),
            "rise_bootstrap": rewards.com_upward_velocity(vz, height, tune.stand_z + tune.rise_margin),
            "gentle_rise": rewards.vertical_accel(az),
            "upright_linear": rewards.upright_linear(cos_tilt),
            "upright_sharp": rewards.gated(
                rewards.upright(tilt, tune.upright_sharp_std), height, tune.prone_z, tune.stand_z
            ),
            "standing_composite": rewards.composite(
                rewards.height_gaussian(height, tune.stand_z, tune.composite_height_std),
                rewards.upright(tilt, tune.composite_upright_std),
                rewards.pose_target(joints[legs], target_legs, tune.composite_pose_std),
            ),
            "joint_torque_rate_l2": rewards.action_rate(force, info["actuator_force"]),
            "body_ang_vel": rewards.body_angular_velocity(sense.world_angular_velocity(data)),
            "angular_momentum": rewards.angular_momentum(sense.read(data, spans.angmom)),
            "action_rate_l2": rewards.action_rate(action, info["last_action"]),
            "self_collisions": rewards.self_collision(
                contact.touching(data, self._rig.self_collision)
            ),
            "neck_action_rate_l2": rewards.action_rate(
                action[self._rig.head_slots], info["last_action"][self._rig.head_slots]
            ),
            "neck_joint_pos_l2": rewards.joint_deviation(
                joints[self._rig.head_slots], self._rig.home[self._rig.head_slots]
            ),
            "joint_torques_l2": rewards.joint_torques(torque),
            "action_over_limit": rewards.action_over_limit(
                raw_target, self._limits, tune.action_overshoot
            ),
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
