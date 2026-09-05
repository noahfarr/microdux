import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground._src import mjx_env

from . import actuator, constants, contact, delay, model, plant, randomize, sense
from . import terrain as rough
from .env import NCONMAX, NJMAX
from .spin import dof

WHEEL_JOINTS = (
    "passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel",
)


class Drive:
    def __init__(
        self,
        variant: str = "rollers",
        ctrl_dt: float = 0.02,
        sim_dt: float = 0.005,
        envs: int = 1,
        impl: str = "jax",
        nconmax: int | None = None,
        njmax: int | None = None,
        stiff: bool = True,
        terrain: rough.Terrain | None = None,
    ):
        self.ctrl_dt = ctrl_dt
        self.sim_dt = sim_dt
        default_nconmax = rough.NCONMAX if terrain is not None else NCONMAX
        default_njmax = rough.NJMAX if terrain is not None else NJMAX
        self.nconmax = nconmax if nconmax is not None else default_nconmax
        self.njmax = njmax if njmax is not None else default_njmax

        iterations = 10 if terrain is None else 30
        ls_iterations = 20 if terrain is None else 50
        self.mj_model, self.layout = model.build(
            variant, timestep=sim_dt, iterations=iterations, ls_iterations=ls_iterations,
            stiff=stiff, terrain=terrain,
        )
        self.mjx_model = model.to_mjx(self.mj_model, impl)
        self.template = mjx_env.make_data(
            self.mj_model, impl=impl,
            naconmax=self.nconmax * envs, njmax=self.njmax,
        )
        self.wiring = plant.wire(self.mj_model, self.layout)
        self.sensors = sense.sensors(self.mj_model)
        self.bam = actuator.load(kp=200.0)
        self.slots = randomize.slots(self.mj_model, self.layout)

        self.home = jnp.asarray(self.layout.home)
        self.foot_sites = np.asarray(self.layout.foot_sites)
        self.stand = jnp.asarray(self.mj_model.key_qpos[self.layout.keyframes["STAND"]])
        self.self_collision = tuple(
            int(i) for i in range(self.mj_model.ngeom) if self.mj_model.geom_contype[i] == 2
        )
        self.feet = (
            contact.subtree(self.mj_model, "ankle_l_v1"),
            contact.subtree(self.mj_model, "ankle_r_v1"),
        )

        names = self.layout.actuators
        self.head_slots = np.array([names.index(j) for j in constants.HEAD_JOINTS])
        self.wheels = np.array([dof(self.mj_model, joint) for joint in WHEEL_JOINTS])

        self.substep_lag = delay.ACTION
        self.joint_vel_line = delay.Line(
            min_lag=delay.JOINT_VEL.min_lag, max_lag=delay.JOINT_VEL.max_lag,
            update_period=0, width=self.mj_model.nu,
        )

    def measure(self, data, info, key, noise):
        spans = self.sensors
        rates = sense.read(data, spans.ang_vel)
        gravity = sense.gravity(data, spans)
        measured = data.qpos[self.wiring.qpos_adr]
        speeds = data.qvel[self.wiring.qvel_adr]

        if info["draw"] is not None:
            measured = measured + info["draw"].encoder_bias
            rates = sense.rotate(info["draw"].imu_quat, rates)
            gravity = sense.rotate(info["draw"].imu_quat, gravity)

        if key is not None and noise is not None:
            rate_key, gravity_key, pos_key, vel_key = jax.random.split(key, 4)
            rates = delay.jitter(rate_key, rates, noise.ang_vel)
            gravity = delay.jitter(gravity_key, gravity, noise.gravity)
            measured = delay.jitter(pos_key, measured, noise.joint_pos)
            speeds = delay.jitter(vel_key, speeds, noise.joint_vel)

        return rates, gravity, measured, speeds

    def lag(self, info, rates, gravity, speeds, key, elapsed):
        if key is None:
            return info, rates, gravity, speeds

        rate_key, gravity_key, vel_key = jax.random.split(key, 3)
        info["joint_vel"], speeds = delay.push(
            info["joint_vel"], self.joint_vel_line, speeds, vel_key, elapsed
        )
        if "imu" in info:
            info["imu"], rates = delay.push(info["imu"], delay.IMU, rates, rate_key, elapsed)
            info["gravity"], gravity = delay.push(
                info["gravity"], delay.GRAVITY, gravity, gravity_key, elapsed
            )
        return info, rates, gravity, speeds

    def shove(self, data, drawn, key, spec):
        kick_key, timer_key = jax.random.split(key)
        due = drawn.push_timer <= 0
        kick = randomize.shove(kick_key, spec)
        qvel = data.qvel.at[:2].add(jnp.where(due, kick, jnp.zeros(2)))
        timer = jnp.where(
            due, randomize.interval(timer_key, spec.push_interval, self.ctrl_dt),
            drawn.push_timer - 1,
        )
        return data.replace(qvel=qvel), drawn.replace(push_timer=timer)

    def plant_model(self, info, com_range, head_com_range):
        if info["draw"] is None:
            return self.mjx_model
        return randomize.apply(
            self.mjx_model, info["draw"], self.slots, com_range, head_com_range,
        )
