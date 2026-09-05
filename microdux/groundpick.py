import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env

from . import constants, contact, curricula, delay, plant, randomize, rewards, sense
from .env import Grounded, PRESERVE


class GroundPick(Grounded):
    def __init__(
        self,
        variant: str = "allcollisions",
        ctrl_dt: float = 0.02,
        sim_dt: float = 0.005,
        episode_length: int = 500,
        action_scale: float = 1.0,
        weights: rewards.GroundPickWeights | None = None,
        tuning: rewards.GroundPickTuning | None = None,
        ranges=None,
        spec: randomize.Spec | None = None,
        noise: delay.Noise | None = None,
        envs: int = curricula.UPSTREAM_ENVS,
        impl: str = "jax",
        nconmax: int | None = None,
        njmax: int | None = None,
        stiff: bool = True,
    ):
        super().__init__(
            variant=variant, ctrl_dt=ctrl_dt, sim_dt=sim_dt, episode_length=episode_length,
            action_scale=action_scale, ranges=ranges, spec=spec, noise=noise, envs=envs,
            impl=impl, nconmax=nconmax, njmax=njmax, stiff=stiff,
        )
        self._weights = weights or rewards.GroundPickWeights()
        self._tuning = tuning or rewards.GroundPickTuning()
        self._limits = jnp.asarray(
            sense.soft_limits(self._mj_model, self._layout, self._tuning.soft_limit_factor)
        )
        self._mouth_site = int(
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, constants.MOUTH_SITE)
        )
        self._jaw_body = int(
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, constants.JAW_BODY)
        )
        self._neck_geoms = contact.subtree(self._mj_model, constants.NECK_BODY)
        self._floor_geom = int(self._layout.floor_geom)

    def _plant_model(self, info, elapsed):
        if info["draw"] is None:
            return self._mjx_model
        return randomize.apply(
            self._mjx_model, info["draw"], self._slots,
            curricula.staircase(elapsed, curricula.GROUND_PICK_COM_RANGE),
            curricula.staircase(elapsed, curricula.HEAD_COM_RANGE),
        )

    def _command_vector(self, info):
        angle = 2.0 * jnp.pi * info["phase"]
        return jnp.concatenate([
            jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0]),
            jnp.zeros(constants.HEAD_COMMAND),
            jnp.zeros(constants.BODY_COMMAND),
        ])

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, pose_key, servo_key, draw_key, phase_key, payload_key, imu_key, vel_key = (
            jax.random.split(rng, 8)
        )

        tune = self._tuning
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
        info = {
            "rng": rng,
            PRESERVE: jnp.zeros((), jnp.int32),
            "servos": servos,
            "gait": contact.rest(len(self._feet)),
            "phase": jax.random.uniform(phase_key, ()),
            "payload": jax.random.uniform(
                payload_key, (), minval=tune.payload_min_kg, maxval=tune.payload_max_kg),
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

    def _payload_wrench(self, data, phase, payload):
        tune = self._tuning
        gate = jnp.clip((phase - tune.hold_end) / tune.payload_ramp, 0.0, 1.0)
        fz = -(gate * payload) * tune.gravity
        p_mouth = data.site_xpos[self._mouth_site]
        p_com = data.xipos[self._jaw_body]
        force = jnp.array([0.0, 0.0, fz])
        torque = jnp.cross(p_mouth - p_com, force)
        return force, torque

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        info["rng"], lag_key, push_key, noise_key, sense_key = jax.random.split(info["rng"], 5)

        elapsed = info[PRESERVE]
        experience = elapsed * self._envs
        lag = jax.random.randint(
            lag_key, (), self._substep_lag.min_lag, self._substep_lag.max_lag + 1
        )

        tune = self._tuning
        phase = (info["phase"] + self._ctrl_dt / tune.period) % 1.0
        info["phase"] = phase

        target = self._home + action * self._action_scale
        if info["draw"] is not None:
            target = target - info["draw"].encoder_bias

        data = state.data
        if info["draw"] is not None:
            data, info["draw"] = self._shove(data, info["draw"], push_key)

        force, torque = self._payload_wrench(data, phase, info["payload"])
        data = data.replace(
            xfrc_applied=data.xfrc_applied.at[self._jaw_body, :3].set(force)
            .at[self._jaw_body, 3:].set(torque)
        )

        data, servos, history = plant.advance(
            self._plant_model(info, experience), data, info["servos"], self._bam,
            target, self._wiring, self.n_substeps,
            history=info["targets"], lag=lag,
        )
        info["servos"] = servos
        info["targets"] = history

        terms = self._rewards(data, info, action, phase)
        weights = dict(vars(self._weights))
        weights["action_rate_l2"] = curricula.staircase(
            experience, curricula.GROUND_PICK_ACTION_RATE_WEIGHT
        )
        total = sum(jnp.nan_to_num(terms[name]) * weights[name] for name in terms)
        reward = total * self._ctrl_dt

        info["previous_action"] = info["last_action"]
        info["last_action"] = action
        info[PRESERVE] = elapsed + 1

        broken = ~jnp.isfinite(data.qpos).all() | ~jnp.isfinite(data.qvel).all()
        done = broken * 1.0

        metrics = dict(state.metrics)
        for name, value in terms.items():
            metrics[f"reward/{name}"] = value

        obs = self._observe(data, info, noise_key, sense_key, elapsed)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _rewards(self, data, info, action, phase):
        spans = self._sensors
        tune = self._tuning
        joints = data.qpos[self._wiring.qpos_adr]
        legs = self._leg_slots
        head = self._head_slots

        mouth_z = data.site_xpos[self._mouth_site, 2]
        mouth_xmat = data.site_xmat[self._mouth_site]
        down_gate = rewards.phase_descend(phase, tune.descent_end, tune.hold_end, tune.rise_end)
        up_gate = rewards.phase_rise(phase, tune.hold_end, tune.rise_end)

        touching = contact.touching(data, self._feet)
        foot_xmat = data.site_xmat[self._foot_sites]
        neck_force = contact.forces(
            self._mjx_model, data, (self._neck_geoms,), partner=self._floor_geom
        )[0]

        return {
            "mouth_ground_proximity": down_gate * rewards.height_gaussian(
                mouth_z, 0.0, tune.mouth_std),
            "mouth_perpendicular": down_gate * rewards.mouth_alignment(mouth_xmat),
            "return_pose_legs": up_gate * rewards.pose_target(
                joints[legs], self._home[legs], tune.return_pose_std),
            "return_pose_neck": up_gate * rewards.pose_target(
                joints[head], self._home[head], tune.return_neck_std),
            "return_upright": up_gate * rewards.upright(
                sense.tilt(data, spans), tune.return_upright_std),
            "neck_vel_descent": (phase < tune.hold_end) * rewards.joint_vel_l2(
                data.qvel[self._wiring.qvel_adr][head]),
            "upright": rewards.upright(sense.tilt(data, spans), tune.upright_std),
            "body_ang_vel": rewards.body_angular_velocity(sense.world_angular_velocity(data)),
            "angular_momentum": rewards.angular_momentum(sense.read(data, spans.angmom)),
            "dof_pos_limits": rewards.joint_pos_limits(joints, self._limits),
            "feet_grounded": rewards.feet_grounded(touching),
            "feet_flat": rewards.feet_flat(foot_xmat, sense.GRAVITY, None),
            "action_rate_l2": rewards.action_rate(action, info["last_action"]),
            "neck_action_rate_l2": rewards.action_rate(
                action[head], info["last_action"][head]),
            "joint_torques_l2": rewards.joint_torques(data.actuator_force),
            "self_collisions": rewards.self_collision(
                contact.touching(data, self._self_collision)),
            "head_impact_penalty": rewards.body_impact(neck_force, tune.head_impact_threshold),
        }
