import math

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from . import (
    actuator, commands, constants, contact, curricula, delay,
    model, plant, randomize, rewards, sense,
)
from . import terrain as rough

TILT_LIMIT = np.radians(70.0)
PRESERVE = "AutoResetWrapper_preserve_info"
NCONMAX = 64
NJMAX = 1500


class Velocity(mjx_env.MjxEnv):
    def __init__(
        self,
        variant: str = "walk",
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
        terrain: rough.Config | None = None,
    ):
        self._envs = envs
        self._terrain = rough.generate(terrain) if terrain is not None else None
        self._nconmax = nconmax if nconmax is not None else (
            rough.NCONMAX if self._terrain is not None else NCONMAX
        )
        self._njmax = njmax if njmax is not None else (
            rough.NJMAX if self._terrain is not None else NJMAX
        )
        self._ctrl_dt = ctrl_dt
        self._sim_dt = sim_dt
        self._episode_length = episode_length
        self._action_scale = action_scale
        self._weights = weights or rewards.Weights()
        self._tuning = tuning or rewards.Tuning()
        self._ranges = ranges or commands.Ranges()
        self._spec = spec
        self._noise = noise

        self._impl = impl
        solver_iterations = 10 if self._terrain is None else 30
        solver_ls_iterations = 20 if self._terrain is None else 50
        self._mj_model, self._layout = model.build(
            variant, timestep=sim_dt, iterations=solver_iterations,
            ls_iterations=solver_ls_iterations, stiff=stiff, terrain=self._terrain,
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
        self._limits = jnp.asarray(
            sense.soft_limits(self._mj_model, self._layout, self._tuning.soft_limit_factor)
        )
        self._feet = tuple(int(g) for g in self._layout.foot_geoms)
        self._foot_sites = np.asarray(self._layout.foot_sites)
        self._stand = jnp.asarray(self._mj_model.key_qpos[self._layout.keyframes["STAND"]])
        self._self_collision = tuple(
            int(i) for i in range(self._mj_model.ngeom) if self._mj_model.geom_contype[i] == 2
        )

        names = self._layout.actuators
        self._head_slots = np.array([names.index(j) for j in constants.HEAD_JOINTS])
        self._leg_slots = np.array(
            [i for i, n in enumerate(names) if "head" not in n and "neck" not in n]
        )
        legs = [names[i] for i in self._leg_slots]
        self._standing_std = jnp.asarray(constants.resolve(rewards.STANDING, legs))
        self._walking_std = jnp.asarray(constants.resolve(rewards.WALKING, legs))

        self._action_line = delay.Line(
            min_lag=delay.ACTION.min_lag, max_lag=delay.ACTION.max_lag,
            update_period=0, width=self._mj_model.nu,
        )
        self._substep_lag = delay.ACTION
        self._joint_vel_line = delay.Line(
            min_lag=delay.JOINT_VEL.min_lag, max_lag=delay.JOINT_VEL.max_lag,
            update_period=0, width=self._mj_model.nu,
        )

        if self._terrain is None:
            self._read_experience = lambda preserved: preserved
            self._read_level = lambda preserved: jnp.zeros((), jnp.int32)
            self._write_preserve = lambda preserved, experience, level: experience
            self._terrain_height = lambda xy, level: jnp.zeros(xy.shape[:-1])
        else:
            self._hfield_variants = jnp.asarray(self._terrain.variants)
            pattern = jnp.asarray(self._terrain.pattern)
            amplitudes = jnp.asarray(self._terrain.amplitudes)
            radius = self._terrain.config.radius

            self._read_experience = lambda preserved: preserved.experience
            self._read_level = lambda preserved: preserved.level
            self._write_preserve = lambda preserved, experience, level: preserved.replace(
                experience=experience, level=level
            )
            self._terrain_height = lambda xy, level: rough.height(
                pattern, radius, amplitudes[level], xy
            )

    @property
    def xml_path(self) -> str:
        return str(constants.XMLS / model.SCENES["walk"])

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

        qpos = self._stand
        if self._spec is not None:
            quat, height = randomize.tilted(pose_key, self._spec, self._spec.base_height)
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
        if self._terrain is None:
            preserved = jnp.zeros((), jnp.int32)
        else:
            preserved = rough.Progress(
                experience=jnp.zeros((), jnp.int32), level=jnp.zeros((), jnp.int32)
            )
        info = {
            "rng": rng,
            PRESERVE: preserved,
            "servos": servos,
            "gait": contact.rest(len(self._feet)),
            "commands": commands.rest(command_key, self._ranges, self._ctrl_dt),
            "last_action": zeros,
            "previous_action": zeros,
            "head_bias": jnp.zeros(len(constants.HEAD_JOINTS)),
            "targets": jnp.tile(self._home, (delay.ACTION.max_lag + 1, 1)),
            "draw": drawn,
        }
        if self._terrain is not None:
            info["terrain_promoted"] = jnp.zeros((), dtype=bool)
            info["terrain_ticks"] = jnp.zeros((), jnp.int32)

        rates, gravity, _, speeds = self._measure(data, info, None)
        info["joint_vel"] = delay.rest(self._joint_vel_line, speeds)
        if self._spec is not None:
            info["imu"] = delay.start(delay.IMU, rates, imu_key)
            info["gravity"] = delay.start(delay.GRAVITY, gravity, vel_key)

        metrics = {f"reward/{name}": jnp.zeros(()) for name in vars(self._weights)}
        obs = self._observe(data, info)
        return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def _plant_model(self, info, elapsed):
        model = self._mjx_model
        if self._terrain is not None:
            level = self._read_level(info[PRESERVE])
            model = model.tree_replace({"hfield_data": self._hfield_variants[level]})
        if info["draw"] is None:
            return model
        return randomize.apply(
            model, info["draw"], self._slots,
            curricula.staircase(elapsed, curricula.COM_RANGE),
            curricula.staircase(elapsed, curricula.HEAD_COM_RANGE),
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        info["rng"], lag_key, push_key, command_key, noise_key, sense_key = jax.random.split(
            info["rng"], 6
        )

        elapsed = self._read_experience(info[PRESERVE])
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

        touching = contact.touching(data, self._feet)
        level = self._read_level(info[PRESERVE])
        foot_xy = data.site_xpos[self._foot_sites][:, :2]
        ground = self._terrain_height(foot_xy, level)
        heights = data.site_xpos[self._foot_sites][:, 2] - ground
        peak = info["gait"].swing_peak
        gait, filtered, landed, air = contact.tally(
            info["gait"], touching, heights, self._ctrl_dt
        )

        head = data.qpos[self._wiring.qpos_adr][self._head_slots]
        head_error = head - (self._home[self._head_slots] + info["commands"].head)
        info["head_bias"] = rewards.blend(
            info["head_bias"], head_error, self._ctrl_dt, self._tuning.head_tau
        )

        terms = self._rewards(data, info, action, touching, air, landed, peak, heights)
        weights = dict(vars(self._weights))
        weights["head_pose_bias"] = curricula.staircase(
            experience, curricula.HEAD_POSE_BIAS_WEIGHT
        )
        weights["action_rate_l2"] = curricula.staircase(
            experience, curricula.ACTION_RATE_WEIGHT
        )
        total = sum(jnp.nan_to_num(terms[name]) * weights[name] for name in terms)
        reward = total * self._ctrl_dt

        info["gait"] = gait
        info["previous_action"] = info["last_action"]
        info["last_action"] = action
        info["commands"] = commands.refresh(
            info["commands"], command_key, self._ranges, self._ctrl_dt,
            head_ranges=curricula.head_ranges(experience),
            standing_fraction=curricula.staircase(experience, curricula.STANDING_FRACTION),
        )

        fell = sense.tilt(data, self._sensors) > np.sin(TILT_LIMIT) ** 2
        broken = ~jnp.isfinite(data.qpos).all() | ~jnp.isfinite(data.qvel).all()

        if self._terrain is None:
            info[PRESERVE] = self._write_preserve(info[PRESERVE], elapsed + 1, level)
            done = (fell | broken) * 1.0
        else:
            distance = jnp.linalg.norm(data.qpos[:2])
            info["terrain_ticks"] = info["terrain_ticks"] + 1
            expected = (
                jnp.linalg.norm(info["commands"].twist[:2])
                * info["terrain_ticks"].astype(jnp.float32) * self._ctrl_dt
                * self._terrain.config.demote_fraction
            )
            next_level, promoted, _, _ = rough.advance(
                level, info["terrain_promoted"], distance,
                self._terrain.promote_radius, fell | broken, expected,
                self._terrain.config.rows,
            )
            info["terrain_promoted"] = promoted
            info[PRESERVE] = self._write_preserve(info[PRESERVE], elapsed + 1, next_level)
            out_of_bounds = distance > self._terrain.bounds_radius
            done = (fell | broken | out_of_bounds) * 1.0

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

    def _rewards(self, data, info, action, touching, air, landed, peak, heights):
        spans = self._sensors
        twist = info["commands"].twist
        joints = data.qpos[self._wiring.qpos_adr]
        velocity = sense.foot_velocity(data, spans)
        tune = self._tuning

        return {
            "track_linear_velocity": rewards.track_linear_velocity(
                sense.root_linear_velocity(data), twist, tune.linear_std),
            "track_angular_velocity": rewards.track_angular_velocity(
                sense.root_angular_velocity(data), twist, tune.angular_std),
            "upright": rewards.upright(sense.tilt(data, spans), tune.upright_std),
            "pose": rewards.variable_posture(
                joints[self._leg_slots], self._home[self._leg_slots], twist,
                self._standing_std, self._walking_std, self._walking_std,
                tune.walking_threshold, tune.running_threshold),
            "body_ang_vel": rewards.body_angular_velocity(sense.world_angular_velocity(data)),
            "angular_momentum": rewards.angular_momentum(sense.read(data, spans.angmom)),
            "dof_pos_limits": rewards.joint_pos_limits(joints, self._limits),
            "action_rate_l2": rewards.action_rate(action, info["last_action"]),
            "air_time": rewards.feet_air_time(
                air, twist, tune.air_time_min, tune.air_time_max, tune.command_threshold),
            "foot_clearance": rewards.feet_clearance(
                heights, velocity, twist, tune.target_height, tune.command_threshold),
            "foot_swing_height": rewards.feet_swing_height(
                peak, landed, twist, tune.target_height, tune.command_threshold),
            "foot_slip": rewards.feet_slip(velocity, touching, twist, tune.command_threshold),
            "self_collisions": rewards.self_collision(
                contact.touching(data, self._self_collision)),
            "head_pose_tracking": rewards.head_pose_tracking(
                joints[self._head_slots], info["commands"].head,
                self._home[self._head_slots], tune.head_std),
            "head_pose_bias": rewards.head_pose_bias(info["head_bias"]),
        }

    def _measure(self, data, info, key):
        spans = self._sensors
        rates = sense.read(data, spans.ang_vel)
        gravity = sense.gravity(data, spans)
        measured = plant.readout(
            data.qpos, self._wiring.qpos_adr,
            self._wiring.backlash_qpos_adr, self._wiring.backlash_mask,
        )
        speeds = plant.readout(
            data.qvel, self._wiring.qvel_adr,
            self._wiring.backlash_qvel_adr, self._wiring.backlash_mask,
        )

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

        state = jnp.concatenate([
            rates,
            gravity,
            measured - self._home,
            speeds,
            info["last_action"],
            commands.vector(info["commands"]),
        ])

        truth = jnp.concatenate([
            sense.read(data, spans.ang_vel),
            sense.gravity(data, spans),
            data.qpos[self._wiring.qpos_adr] - self._home,
            data.qvel[self._wiring.qvel_adr],
            info["last_action"],
            commands.vector(info["commands"]),
        ])

        level = self._read_level(info[PRESERVE])
        foot_xy = data.site_xpos[self._foot_sites][:, :2]
        ground = self._terrain_height(foot_xy, level)
        privileged = jnp.concatenate([
            truth,
            sense.root_linear_velocity(data),
            data.site_xpos[self._foot_sites][:, 2] - ground,
            info["gait"].air_time,
            info["gait"].last_contact * 1.0,
            contact.forces(self._mjx_model, data, self._feet).reshape(-1),
        ])
        return {"state": state, "privileged_state": privileged}


class Grounded(mjx_env.MjxEnv):
    def __init__(
        self,
        variant: str = "allcollisions",
        ctrl_dt: float = 0.02,
        sim_dt: float = 0.005,
        episode_length: int = 300,
        action_scale: float = 1.0,
        ranges: commands.Ranges | None = None,
        spec: randomize.Spec | None = None,
        noise: delay.Noise | None = None,
        envs: int = curricula.UPSTREAM_ENVS,
        impl: str = "jax",
        nconmax: int | None = None,
        njmax: int | None = None,
        stiff: bool = True,
        iterations: int = 10,
        ls_iterations: int = 20,
    ):
        self._envs = envs
        self._nconmax = nconmax if nconmax is not None else NCONMAX
        self._njmax = njmax if njmax is not None else NJMAX
        self._ctrl_dt = ctrl_dt
        self._sim_dt = sim_dt
        self._episode_length = episode_length
        self._action_scale = action_scale
        self._ranges = ranges or commands.Ranges()
        self._spec = spec
        self._noise = noise

        self._impl = impl
        self._mj_model, self._layout = model.build(
            variant, timestep=sim_dt, iterations=iterations, ls_iterations=ls_iterations, stiff=stiff
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
        self._stand = jnp.asarray(self._mj_model.key_qpos[self._layout.keyframes["STAND"]])
        self._self_collision = tuple(
            int(i) for i in range(self._mj_model.ngeom) if self._mj_model.geom_contype[i] == 2
        )

        names = self._layout.actuators
        self._head_slots = np.array([names.index(j) for j in constants.HEAD_JOINTS])
        self._leg_slots = np.array(
            [i for i, n in enumerate(names) if "head" not in n and "neck" not in n]
        )
        self._sit = jnp.asarray(constants.resolve(constants.SIT, names))

        self._action_line = delay.Line(
            min_lag=delay.ACTION.min_lag, max_lag=delay.ACTION.max_lag,
            update_period=0, width=self._mj_model.nu,
        )
        self._substep_lag = delay.ACTION
        self._joint_vel_line = delay.Line(
            min_lag=delay.JOINT_VEL.min_lag, max_lag=delay.JOINT_VEL.max_lag,
            update_period=0, width=self._mj_model.nu,
        )

    @property
    def xml_path(self) -> str:
        return str(constants.XMLS / model.SCENES["allcollisions"])

    @property
    def action_size(self) -> int:
        return self._mj_model.nu

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model

    def _plant_model(self, info, elapsed):
        if info["draw"] is None:
            return self._mjx_model
        return randomize.apply(
            self._mjx_model, info["draw"], self._slots,
            curricula.staircase(elapsed, curricula.COM_RANGE),
            curricula.staircase(elapsed, curricula.HEAD_COM_RANGE),
        )

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

    def _measure(self, data, info, key):
        spans = self._sensors
        rates = sense.read(data, spans.ang_vel)
        gravity = sense.gravity(data, spans)
        measured = plant.readout(
            data.qpos, self._wiring.qpos_adr,
            self._wiring.backlash_qpos_adr, self._wiring.backlash_mask,
        )
        speeds = plant.readout(
            data.qvel, self._wiring.qvel_adr,
            self._wiring.backlash_qvel_adr, self._wiring.backlash_mask,
        )

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

    def _command_vector(self, info):
        return commands.vector(info["commands"])

    def _observe(self, data, info, key=None, lag_key=None, elapsed=0):
        spans = self._sensors
        rates, gravity, measured, speeds = self._measure(data, info, key)
        info, rates, gravity, speeds = self._lag(
            info, rates, gravity, speeds, lag_key, elapsed
        )
        command_vector = self._command_vector(info)

        state = jnp.concatenate([
            rates,
            gravity,
            measured - self._home,
            speeds,
            info["last_action"],
            command_vector,
        ])

        truth = jnp.concatenate([
            sense.read(data, spans.ang_vel),
            sense.gravity(data, spans),
            data.qpos[self._wiring.qpos_adr] - self._home,
            data.qvel[self._wiring.qvel_adr],
            info["last_action"],
            command_vector,
        ])

        privileged = jnp.concatenate([
            truth,
            sense.root_linear_velocity(data),
            data.site_xpos[self._foot_sites][:, 2],
            info["gait"].air_time,
            info["gait"].last_contact * 1.0,
            contact.forces(self._mjx_model, data, self._feet).reshape(-1),
        ])
        return {"state": state, "privileged_state": privileged}


class StandUp(Grounded):
    def __init__(
        self,
        variant: str = "allcollisions",
        ctrl_dt: float = 0.02,
        sim_dt: float = 0.005,
        episode_length: int = 300,
        action_scale: float = 1.0,
        weights: rewards.StandUpWeights | None = None,
        tuning: rewards.StandUpTuning | None = None,
        ranges: commands.Ranges | None = None,
        spec: randomize.Spec | None = None,
        noise: delay.Noise | None = None,
        envs: int = curricula.UPSTREAM_ENVS,
        impl: str = "jax",
        nconmax: int | None = None,
        njmax: int | None = None,
        stiff: bool = True,
        ground_mix: randomize.GroundMix | None = None,
    ):
        super().__init__(
            variant=variant, ctrl_dt=ctrl_dt, sim_dt=sim_dt, episode_length=episode_length,
            action_scale=action_scale, ranges=ranges, spec=spec, noise=noise, envs=envs,
            impl=impl, nconmax=nconmax, njmax=njmax, stiff=stiff,
        )
        self._weights = weights or rewards.StandUpWeights()
        self._tuning = tuning or rewards.StandUpTuning()
        self._limits = jnp.asarray(
            sense.soft_limits(self._mj_model, self._layout, self._tuning.soft_limit_factor)
        )
        self._ground_mix = ground_mix or randomize.GroundMix()

    def respawn(self, state, preserved, rng):
        weights = curricula.ranges(preserved * self._envs, curricula.GROUND_MIX)
        return self.reset(rng, self._ground_mix.replace(
            standing=weights[0], sitting=weights[1],
            face_down=weights[2], face_up=weights[3],
        ))

    def reset(self, rng: jax.Array, mix=None) -> mjx_env.State:
        rng, command_key, draw_key, servo_key, ground_key, imu_key, vel_key = jax.random.split(rng, 7)

        joints, quat, height = randomize.groundstate(
            ground_key, self._ground_mix if mix is None else mix, self._home, self._sit
        )
        qpos = self._stand.at[2].set(height).at[3:7].set(quat).at[self._wiring.qpos_adr].set(joints)

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
            "head_bias": jnp.zeros(len(constants.HEAD_JOINTS)),
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

        terms = self._rewards(data, info, action)
        weights = dict(vars(self._weights))
        weights["head_pose_bias"] = curricula.staircase(
            experience, curricula.HEAD_POSE_BIAS_WEIGHT
        )
        weights["action_rate_l2"] = curricula.staircase(
            experience, curricula.ACTION_RATE_WEIGHT
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

    def _rewards(self, data, info, action):
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

        return {
            "pose_legs": rewards.pose_target(joints[legs], target_legs, tune.pose_std),
            "pose_legs_l1": rewards.pose_l1(joints[legs], target_legs),
            "head_pose_tracking": rewards.head_pose_tracking(
                joints[self._head_slots], info["commands"].head,
                self._home[self._head_slots], tune.head_std),
            "head_pose_bias": rewards.head_pose_bias(info["head_bias"]),
            "height": rewards.height_gaussian(height, self._stand[2], tune.height_std),
            "height_sharp": rewards.height_gaussian(height, self._stand[2], tune.height_sharp_std),
            "height_l1": rewards.height_l1(height, self._stand[2]),
            "rise_bootstrap": rewards.com_upward_velocity(
                vz, height, self._stand[2] + tune.rise_margin),
            "gentle_rise": rewards.vertical_accel(az),
            "upright_linear": rewards.upright_linear(cos_tilt),
            "upright_sharp": rewards.gated(
                rewards.upright(tilt, tune.upright_sharp_std), height, constants.SIT_Z,
                self._stand[2]),
            "standing_composite": rewards.composite(
                rewards.height_gaussian(height, self._stand[2], tune.composite_height_std),
                rewards.upright(tilt, tune.composite_upright_std),
                rewards.pose_target(joints[legs], target_legs, tune.composite_pose_std),
            ),
            "body_ang_vel": rewards.body_angular_velocity(sense.world_angular_velocity(data)),
            "angular_momentum": rewards.angular_momentum(sense.read(data, spans.angmom)),
            "dof_pos_limits": rewards.joint_pos_limits(joints, self._limits),
            "action_rate_l2": rewards.action_rate(action, info["last_action"]),
            "self_collisions": rewards.self_collision(
                contact.touching(data, self._self_collision)),
        }


class SitStand(Grounded):
    def __init__(
        self,
        variant: str = "allcollisions",
        ctrl_dt: float = 0.02,
        sim_dt: float = 0.005,
        episode_length: int = 600,
        action_scale: float = 1.0,
        weights: rewards.SitStandWeights | None = None,
        tuning: rewards.SitStandTuning | None = None,
        ranges: commands.Ranges | None = None,
        spec: randomize.Spec | None = None,
        noise: delay.Noise | None = None,
        envs: int = curricula.UPSTREAM_ENVS,
        impl: str = "jax",
        nconmax: int | None = 200,
        njmax: int | None = None,
        stiff: bool = True,
        ground_mix: randomize.GroundMix | None = None,
        sit_prob: float = 0.5,
        dwell: tuple = (3.5, 6.5),
        ramp_s: float = 2.0,
    ):
        super().__init__(
            variant=variant, ctrl_dt=ctrl_dt, sim_dt=sim_dt, episode_length=episode_length,
            action_scale=action_scale, ranges=ranges, spec=spec, noise=noise, envs=envs,
            impl=impl, nconmax=nconmax, njmax=njmax, stiff=stiff,
            iterations=30, ls_iterations=50,
        )
        self._weights = weights or rewards.SitStandWeights()
        self._tuning = tuning or rewards.SitStandTuning()
        self._limits = jnp.asarray(
            sense.soft_limits(self._mj_model, self._layout, self._tuning.soft_limit_factor)
        )
        self._ground_mix = ground_mix or randomize.GroundMix(
            standing=0.5, sitting=0.5, face_down=0.0, face_up=0.0,
            standing_height=(0.11, 0.12), sitting_height=(0.06, 0.075), sitting_tilt=8.0,
            sitting_noise=0.10,
        )
        self._sit_prob = sit_prob
        self._dwell = dwell
        self._ramp_s = ramp_s
        tune = self._tuning
        self._stillness_tilt_full = math.sin(math.radians(tune.stillness_tilt_full_deg)) ** 2
        self._stillness_tilt_zero = math.sin(math.radians(tune.stillness_tilt_zero_deg)) ** 2

    def _target(self, blend):
        joints = self._home + blend * (self._sit - self._home)
        height = self._stand[2] + blend * (constants.SIT_Z - self._stand[2])
        return joints, height

    def _command_vector(self, info):
        twist = jnp.array([info["posture"].blend, 0.0, 0.0])
        return jnp.concatenate([twist, info["commands"].head, jnp.zeros(constants.BODY_COMMAND)])

    def respawn(self, state, preserved, rng):
        weights = curricula.ranges(preserved * self._envs, curricula.GROUND_MIX)
        return self.reset(rng, self._ground_mix.replace(
            standing=weights[0], sitting=weights[1],
            face_down=weights[2], face_up=weights[3],
        ))

    def reset(self, rng: jax.Array, mix=None) -> mjx_env.State:
        rng, command_key, draw_key, servo_key, ground_key, posture_key, imu_key, vel_key = (
            jax.random.split(rng, 8)
        )

        joints, quat, height = randomize.groundstate(
            ground_key, self._ground_mix if mix is None else mix, self._home, self._sit
        )
        qpos = self._stand.at[2].set(height).at[3:7].set(quat).at[self._wiring.qpos_adr].set(joints)

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
            "posture": commands.rest_posture(posture_key, self._sit_prob, self._dwell, self._ctrl_dt),
            "last_action": zeros,
            "previous_action": zeros,
            "head_bias": jnp.zeros(len(constants.HEAD_JOINTS)),
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

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        info["rng"], lag_key, push_key, command_key, posture_key, noise_key, sense_key = (
            jax.random.split(info["rng"], 7)
        )

        elapsed = info[PRESERVE]
        experience = elapsed * self._envs
        lag = jax.random.randint(
            lag_key, (), self._substep_lag.min_lag, self._substep_lag.max_lag + 1
        )

        info["posture"] = commands.refresh_posture(
            info["posture"], posture_key, self._sit_prob, self._dwell, self._ctrl_dt, self._ramp_s
        )
        target_joints, target_height = self._target(info["posture"].blend)

        target = target_joints + action * self._action_scale
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

        terms = self._rewards(data, info, action, target_joints, target_height)
        weights = dict(vars(self._weights))
        weights["action_rate_l2"] = curricula.staircase(
            experience, curricula.ACTION_RATE_WEIGHT
        )
        weights["rise_speed"] = curricula.staircase(
            experience, curricula.SITSTAND_RISE_SPEED_WEIGHT
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

    def _rewards(self, data, info, action, target_joints, target_height):
        spans = self._sensors
        tune = self._tuning
        joints = data.qpos[self._wiring.qpos_adr]
        legs = self._leg_slots
        target_legs = target_joints[legs]

        height = data.qpos[2]
        vz = data.qvel[2]
        az = data.qacc[2]
        cos_tilt = -sense.gravity(data, spans)[2]
        tilt = sense.tilt(data, spans)
        standing = info["posture"].blend < 0.5
        height_error = jnp.abs(height - target_height)

        head_factor = rewards.head_pose_tracking(
            joints[self._head_slots], info["commands"].head,
            self._home[self._head_slots], tune.head_std)

        return {
            "posture_pose_legs": rewards.pose_target(joints[legs], target_legs, tune.pose_std),
            "posture_pose_l1": rewards.pose_l1(joints[legs], target_legs),
            "head_pose_tracking": head_factor,
            "posture_height": rewards.height_gaussian(height, target_height, tune.height_std),
            "posture_height_sharp": rewards.height_gaussian(
                height, target_height, tune.height_sharp_std),
            "posture_height_l1": rewards.height_l1(height, target_height),
            "rise_bootstrap": jnp.where(
                standing, rewards.com_upward_velocity(
                    vz, height, self._stand[2] + tune.rise_margin), 0.0),
            "descent_speed": rewards.downward_speed(vz, tune.max_descent_speed),
            "rise_speed": rewards.upward_speed(vz, tune.max_rise_speed),
            "gentle_motion": rewards.vertical_accel(az),
            "upright_linear": rewards.upright_linear(cos_tilt),
            "upright_while_tall": rewards.upright_linear(cos_tilt) * rewards.gate(
                height, tune.sit_upright_z, tune.stand_upright_z),
            "posture_stillness": rewards.settled(
                height_error, tilt, jnp.abs(vz),
                tune.stillness_band_zero, tune.stillness_band_full,
                self._stillness_tilt_zero, self._stillness_tilt_full, tune.stillness_vel_std),
            "posture_composite": rewards.composite(
                rewards.height_gaussian(height, target_height, tune.composite_height_std),
                rewards.upright(tilt, tune.composite_upright_std),
                rewards.pose_target(joints[legs], target_legs, tune.composite_pose_std),
                head_factor,
            ),
            "body_ang_vel": rewards.body_angular_velocity(sense.world_angular_velocity(data)),
            "angular_momentum": rewards.angular_momentum(sense.read(data, spans.angmom)),
            "dof_pos_limits": rewards.joint_pos_limits(joints, self._limits),
            "action_rate_l2": rewards.action_rate(action, info["last_action"]),
            "self_collisions": rewards.self_collision(
                contact.touching(data, self._self_collision)),
        }
