import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from . import actuator, commands, constants, contact, curricula, delay, model, plant, randomize, rewards, sense
from .env import Grounded, NCONMAX, NJMAX, PRESERVE, TILT_LIMIT

model.SCENES.setdefault("ball", "scene_ball.xml")

RANGES = commands.Ranges(
    lin_x=(-0.01, 0.01), lin_y=(-0.01, 0.01), ang_z=(-0.05, 0.05),
    head=((0.0, 0.0),) * 4, body=((0.0, 0.0),) * 6,
)


class BallKick(Grounded):
    def __init__(
        self,
        kick_foot: str = "right",
        ctrl_dt: float = 0.02,
        sim_dt: float = 0.005,
        episode_length: int = 250,
        action_scale: float = 1.0,
        weights: rewards.BallKickWeights | None = None,
        tuning: rewards.BallKickTuning | None = None,
        ranges: commands.Ranges | None = None,
        spec: randomize.Spec | None = None,
        noise: delay.Noise | None = None,
        envs: int = curricula.UPSTREAM_ENVS,
        impl: str = "jax",
        nconmax: int | None = 50,
        njmax: int | None = None,
        stiff: bool = True,
        iterations: int = 10,
        ls_iterations: int = 20,
    ):
        assert kick_foot in ("left", "right")
        self._kick_foot = kick_foot
        self._support_index = 0 if kick_foot == "right" else 1
        self._kick_offset_sign = -1.0 if kick_foot == "right" else 1.0

        self._envs = envs
        self._nconmax = nconmax if nconmax is not None else NCONMAX
        self._njmax = njmax if njmax is not None else NJMAX
        self._ctrl_dt = ctrl_dt
        self._sim_dt = sim_dt
        self._episode_length = episode_length
        self._action_scale = action_scale
        self._ranges = ranges or RANGES
        self._spec = spec
        self._noise = noise

        self._impl = impl
        self._mj_model, self._layout = model.build(
            "ball", timestep=sim_dt, iterations=iterations, ls_iterations=ls_iterations,
            stiff=stiff,
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
        self._feet = tuple(int(g) for g in self._layout.foot_geoms)
        self._foot_sites = np.asarray(self._layout.foot_sites)
        self._self_collision = tuple(
            int(i) for i in range(self._mj_model.ngeom) if self._mj_model.geom_contype[i] == 2
        )

        names = self._layout.actuators
        self._head_slots = np.array([names.index(j) for j in constants.HEAD_JOINTS])
        self._leg_slots = np.array(
            [i for i, n in enumerate(names) if "head" not in n and "neck" not in n]
        )

        self._action_line = delay.Line(
            min_lag=delay.ACTION.min_lag, max_lag=delay.ACTION.max_lag,
            update_period=0, width=self._mj_model.nu,
        )
        self._substep_lag = delay.ACTION
        self._joint_vel_line = delay.Line(
            min_lag=delay.JOINT_VEL.min_lag, max_lag=delay.JOINT_VEL.max_lag,
            update_period=0, width=self._mj_model.nu,
        )

        self._weights = weights or rewards.BallKickWeights()
        self._tuning = tuning or rewards.BallKickTuning()
        self._limits = jnp.asarray(
            sense.soft_limits(self._mj_model, self._layout, self._tuning.soft_limit_factor)
        )

        root_joint = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, constants.ROOT_JOINT)
        self._root_qpos_adr = int(self._mj_model.jnt_qposadr[root_joint])

        ball_joint = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, constants.BALL_FREE_JOINT
        )
        self._ball_qpos_adr = int(self._mj_model.jnt_qposadr[ball_joint])
        self._ball_qvel_adr = int(self._mj_model.jnt_dofadr[ball_joint])
        self._support_foot_geom = self._feet[self._support_index]

    @property
    def xml_path(self) -> str:
        return str(constants.XMLS / model.SCENES["ball"])

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, pose_key, ball_key, joint_key, servo_key, draw_key, command_key, imu_key, vel_key = (
            jax.random.split(rng, 9)
        )

        tune = self._tuning
        quat, height = randomize.tilted(
            pose_key, self._spec or randomize.Spec(), tune.standing_z,
            pitch_range=(-tune.standing_tilt_deg, tune.standing_tilt_deg),
            roll_range=(-tune.standing_tilt_deg, tune.standing_tilt_deg),
        )
        joints = self._home + jax.random.uniform(
            joint_key, self._home.shape, minval=-tune.joint_noise, maxval=tune.joint_noise,
        )

        yaw = sense.yaw(quat)
        rotation = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw)], [jnp.sin(yaw), jnp.cos(yaw)]])
        offset = jnp.array(
            [tune.offset_x, self._kick_offset_sign * tune.offset_abs_y]
        ) + jax.random.uniform(ball_key, (2,), minval=-tune.offset_noise, maxval=tune.offset_noise)
        ball_xy = rotation @ offset
        kick_dir = jnp.array([jnp.cos(yaw), jnp.sin(yaw)])

        qpos = jnp.zeros(self._mj_model.nq)
        qpos = qpos.at[self._root_qpos_adr:self._root_qpos_adr + 3].set(
            jnp.array([0.0, 0.0, height])
        )
        qpos = qpos.at[self._root_qpos_adr + 3:self._root_qpos_adr + 7].set(quat)
        qpos = qpos.at[self._wiring.qpos_adr].set(joints)
        qpos = qpos.at[self._ball_qpos_adr:self._ball_qpos_adr + 2].set(ball_xy)
        qpos = qpos.at[self._ball_qpos_adr + 2].set(tune.ball_radius)
        qpos = qpos.at[self._ball_qpos_adr + 3].set(1.0)

        data = self._template.replace(qpos=qpos, qvel=jnp.zeros(self._mj_model.nv))
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
            "targets": jnp.tile(self._home, (delay.ACTION.max_lag + 1, 1)),
            "draw": drawn,
            "kick_dir": kick_dir,
        }

        rates, gravity, _, speeds = self._measure(data, info, None)
        info["joint_vel"] = delay.rest(self._joint_vel_line, speeds)
        if self._spec is not None:
            info["imu"] = delay.start(delay.IMU, rates, imu_key)
            info["gravity"] = delay.start(delay.GRAVITY, gravity, vel_key)

        metrics = {f"reward/{name}": jnp.zeros(()) for name in vars(self._weights)}
        obs = self._observe(data, info)
        return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

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

        terms = self._rewards(data, info, action)
        weights = dict(vars(self._weights))
        weights["action_rate_l2"] = curricula.staircase(experience, curricula.ACTION_RATE_WEIGHT)
        total = sum(jnp.nan_to_num(terms[name]) * weights[name] for name in terms)
        reward = total * self._ctrl_dt

        info["previous_action"] = info["last_action"]
        info["last_action"] = action
        info[PRESERVE] = elapsed + 1
        info["commands"] = commands.refresh(
            info["commands"], command_key, self._ranges, self._ctrl_dt,
        )

        spans = self._sensors
        fell = sense.tilt(data, spans) > np.sin(TILT_LIMIT) ** 2
        broken = ~jnp.isfinite(data.qpos).all() | ~jnp.isfinite(data.qvel).all()
        done = (fell | broken) * 1.0

        metrics = dict(state.metrics)
        for name, value in terms.items():
            metrics[f"reward/{name}"] = value

        obs = self._observe(data, info, noise_key, sense_key, elapsed)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _rewards(self, data, info, action):
        spans = self._sensors
        tune = self._tuning
        joints = data.qpos[self._wiring.qpos_adr]
        legs = self._leg_slots
        head = self._head_slots

        ball_vel = data.qvel[self._ball_qvel_adr:self._ball_qvel_adr + 2]
        touching = contact.touching(data, (self._support_foot_geom,))
        height = data.qpos[self._root_qpos_adr + 2]
        tilt = sense.tilt(data, spans)

        return {
            "ball_forward_velocity": rewards.ball_forward_velocity(
                ball_vel, info["kick_dir"], tune.target_speed),
            "ball_speed_overshoot": rewards.ball_speed_overshoot(
                ball_vel, info["kick_dir"], tune.target_speed, tune.overshoot_cap),
            "support_foot_grounded": rewards.feet_grounded(touching),
            "pose_stand_legs": rewards.pose_target(joints[legs], self._home[legs], tune.pose_std),
            "pose_stand_neck": rewards.pose_target(joints[head], self._home[head], tune.neck_std),
            "upright": rewards.upright(tilt, tune.upright_std),
            "height_stand": rewards.height_gaussian(height, tune.stand_height, tune.height_std),
            "body_ang_vel": rewards.body_angular_velocity(sense.world_angular_velocity(data)),
            "angular_momentum": rewards.angular_momentum(sense.read(data, spans.angmom)),
            "dof_pos_limits": rewards.joint_pos_limits(joints, self._limits),
            "action_rate_l2": rewards.action_rate(action, info["last_action"]),
            "self_collisions": rewards.self_collision(
                contact.touching(data, self._self_collision)),
        }

    def _observe(self, data, info, key=None, lag_key=None, elapsed=0):
        obs = super()._observe(data, info, key, lag_key, elapsed)
        root_pos = data.qpos[self._root_qpos_adr:self._root_qpos_adr + 3]
        root_quat = data.qpos[self._root_qpos_adr + 3:self._root_qpos_adr + 7]
        ball_pos = data.qpos[self._ball_qpos_adr:self._ball_qpos_adr + 3]
        ball_vel = data.qvel[self._ball_qvel_adr:self._ball_qvel_adr + 3]
        pos_local = sense.rotate_inverse(root_quat, ball_pos - root_pos)
        vel_local = sense.rotate_inverse(root_quat, ball_vel)
        privileged = jnp.concatenate([obs["privileged_state"], pos_local, vel_local])
        return {"state": obs["state"], "privileged_state": privileged}
