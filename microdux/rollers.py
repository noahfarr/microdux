import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from . import actuator, commands, constants, contact, curricula, delay, model, plant, randomize, rewards, sense
from .env import NCONMAX, NJMAX, PRESERVE, TILT_LIMIT
from .spin import dof

LIN_X_RANGE = (-0.5, 0.6)
ZERO_HEAD = ((0.0, 0.0),) * 4
ZERO_BODY = ((0.0, 0.0),) * 6


class Rollers(mjx_env.MjxEnv):
    def __init__(
        self,
        ctrl_dt: float = 0.02,
        sim_dt: float = 0.005,
        episode_length: int = 1000,
        action_scale: float = 1.0,
        weights: rewards.RollerWeights | None = None,
        tuning: rewards.RollerTuning | None = None,
        ranges: commands.Ranges | None = None,
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
        self._weights = weights or rewards.RollerWeights()
        self._tuning = tuning or rewards.RollerTuning()
        self._ranges = ranges or commands.Ranges(
            lin_x=LIN_X_RANGE, lin_y=(0.0, 0.0), ang_z=(0.0, 0.0),
            head=ZERO_HEAD, body=ZERO_BODY,
            standing_fraction=0.0, turn_fraction=0.0,
        )
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
        self._limits = jnp.asarray(self._mj_model.jnt_range[self._layout.actuated_joint_ids])
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
        self._leg_slots = np.array(
            [i for i, n in enumerate(names) if "head" not in n and "neck" not in n]
        )
        legs = [names[i] for i in self._leg_slots]
        self._standing_std = jnp.asarray(constants.resolve(rewards.ROLLER_STANDING, legs))
        self._walking_std = jnp.asarray(constants.resolve(rewards.ROLLER_WALKING, legs))
        self._running_std = jnp.asarray(constants.resolve(rewards.ROLLER_RUNNING, legs))
        self._hip_roll_slots = np.array([i for i, n in enumerate(names) if "hip_roll" in n])

        self._wheel_lf = dof(self._mj_model, "passive_LF_wheel")
        self._wheel_lr = dof(self._mj_model, "passive_LR_wheel")
        self._wheel_rf = dof(self._mj_model, "passive_RF_wheel")
        self._wheel_rr = dof(self._mj_model, "passive_RR_wheel")
        self._wheel_omega_scale = self._tuning.wheel_vel_scale / self._tuning.wheel_radius

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
        rng, command_key, draw_key, servo_key, pose_key, imu_key, vel_key = jax.random.split(rng, 7)

        qpos = self._stand.at[2].set(sum(constants.ROLLER_HEIGHT) / 2)
        if self._spec is not None:
            quat, height = randomize.tilted(pose_key, self._spec, constants.ROLLER_HEIGHT)
            qpos = qpos.at[2].set(height).at[3:7].set(quat)

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
            "heading_ref": sense.yaw(sense.root_quat(data)),
            "swing_accum": jnp.zeros(2),
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
        info["rng"], lag_key, push_key, command_key, noise_key, sense_key = jax.random.split(
            info["rng"], 6
        )

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
        gait, filtered, landed, air = contact.tally(
            info["gait"], touching, heights, self._ctrl_dt
        )
        info["swing_accum"] = info["swing_accum"] + (air > 0.0) * self._ctrl_dt

        terms = self._rewards(data, info, action, touching, air, gait, raw_target)
        weights = dict(vars(self._weights))
        weights["action_rate_l2"] = curricula.staircase(experience, curricula.ROLLER_ACTION_RATE_WEIGHT)
        total = sum(jnp.nan_to_num(terms[name]) * weights[name] for name in terms)
        reward = total * self._ctrl_dt

        info["gait"] = gait
        info["commands"] = commands.refresh(info["commands"], command_key, self._ranges, self._ctrl_dt)
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

    def _rewards(self, data, info, action, touching, air, gait, raw_target):
        spans = self._sensors
        joints = data.qpos[self._wiring.qpos_adr]
        tune = self._tuning

        cmd_x = info["commands"].twist[0]
        twist = jnp.array([cmd_x, 0.0, 0.0])
        yaw = sense.yaw(sense.root_quat(data))
        lean = sense.gravity(data, spans)[0]
        fwd_vel = sense.root_linear_velocity(data)[0]

        xmat = data.site_xmat[self._foot_sites]
        torque = data.qfrc_actuator[self._wiring.qvel_adr]
        leg_vel = data.qvel[self._wiring.qvel_adr][self._leg_slots]

        omega_left = (data.qvel[self._wheel_lf] + data.qvel[self._wheel_lr]) / 2.0
        omega_right = (data.qvel[self._wheel_rf] + data.qvel[self._wheel_rr]) / 2.0
        forward_omega = (omega_left + omega_right) / 2.0

        skating_gate = rewards.forward_gate(fwd_vel, tune.skating_vel_gate)
        single_support_gate = rewards.forward_gate(fwd_vel, tune.single_support_vel_gate)
        glide_gate = rewards.forward_gate(fwd_vel, tune.glide_vel_ref)

        return {
            "pose": rewards.variable_posture(
                joints[self._leg_slots], self._home[self._leg_slots], twist,
                self._standing_std, self._walking_std, self._running_std,
                tune.walking_threshold, tune.running_threshold,
            ),
            "upright": rewards.upright(sense.tilt(data, spans), tune.upright_std),
            "body_ang_vel": rewards.body_angular_velocity(sense.world_angular_velocity(data)),
            "angular_momentum": rewards.angular_momentum(sense.read(data, spans.angmom)),
            "action_rate_l2": rewards.action_rate(action, info["last_action"]),
            "com_height_target": rewards.com_height_target(
                data.qpos[2], tune.com_height_min, tune.com_height_max
            ),
            "self_collisions": rewards.self_collision(contact.touching(data, self._self_collision)),
            "feet_flat": rewards.feet_flat(xmat, sense.GRAVITY, touching),
            "neck_action_rate_l2": rewards.action_rate(
                action[self._head_slots], info["last_action"][self._head_slots]
            ),
            "neck_joint_pos_l2": rewards.joint_deviation(
                joints[self._head_slots], self._home[self._head_slots]
            ),
            "joint_torques_l2": rewards.joint_torques(torque),
            "action_over_limit": rewards.action_over_limit(
                raw_target, self._limits, tune.action_overshoot
            ),
            "hip_roll_neutral": rewards.hip_roll_neutral(
                joints[self._hip_roll_slots], self._home[self._hip_roll_slots]
            ),
            "wheel_speed": rewards.wheel_speed(
                cmd_x, forward_omega, self._wheel_omega_scale, False
            ),
            "braking": rewards.braking(cmd_x, fwd_vel, tune.braking_std),
            "skating_air_time": rewards.skating_air_time(
                air, tune.skating_air_min, tune.skating_air_max, cmd_x, skating_gate
            ),
            "glide": rewards.glide(
                gait.contact_time, leg_vel, tune.glide_stillness_std, glide_gate, cmd_x
            ),
            "single_support": rewards.single_support(
                gait.contact_time, cmd_x, single_support_gate, tune.single_support_double_penalty
            ),
            "gait_symmetry": rewards.gait_symmetry(info["swing_accum"]),
            "forward_lean": rewards.forward_lean(
                cmd_x, lean, tune.forward_lean_target, tune.forward_lean_std
            ),
            "heading_hold": rewards.heading_hold(yaw, info["heading_ref"], tune.heading_hold_std),
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

        command = commands.vector(info["commands"])

        state = jnp.concatenate([
            rates,
            gravity,
            measured - self._home,
            speeds,
            info["last_action"],
            command,
        ])

        truth = jnp.concatenate([
            sense.read(data, spans.ang_vel),
            sense.gravity(data, spans),
            data.qpos[self._wiring.qpos_adr] - self._home,
            data.qvel[self._wiring.qvel_adr],
            info["last_action"],
            command,
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
