import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import struct

XL330_ENCODER_COUNTS_PER_REV = 4096
XL330_KP_DIVISOR = 256
XL330_PWM_LIMIT = 885

XL330 = dict(
    vin=7.5,
    kp=400.0,
    error_gain=(XL330_ENCODER_COUNTS_PER_REV / (2 * jnp.pi))
    / (XL330_KP_DIVISOR * XL330_PWM_LIMIT),
    max_pwm=1.0,
    max_current=1.75,
)

FLAGS = {
    "m1": dict(load_dependent=False, directional=False, stribeck=False, quadratic=False),
    "m2": dict(load_dependent=False, directional=False, stribeck=True, quadratic=False),
    "m3": dict(load_dependent=True, directional=False, stribeck=False, quadratic=False),
    "m4": dict(load_dependent=True, directional=False, stribeck=True, quadratic=False),
    "m5": dict(load_dependent=True, directional=True, stribeck=True, quadratic=False),
    "m6": dict(load_dependent=True, directional=True, stribeck=True, quadratic=True),
}


@struct.dataclass
class Bam:
    kt: float
    resistance: float
    armature: float
    kp: float
    error_gain: float
    max_pwm: float
    max_current: float
    friction_base: float
    friction_viscous: float
    friction_stribeck: float
    dtheta_stribeck: float
    alpha: float
    load_friction_base: float
    load_friction_stribeck: float
    load_friction_motor: float
    load_friction_external: float
    load_friction_motor_stribeck: float
    load_friction_external_stribeck: float
    load_friction_motor_quad: float
    load_friction_external_quad: float
    load_dependent: bool = struct.field(pytree_node=False)
    directional: bool = struct.field(pytree_node=False)
    stribeck: bool = struct.field(pytree_node=False)
    quadratic: bool = struct.field(pytree_node=False)
    limited: bool = struct.field(pytree_node=False)


def load(motor: str = "xl330", model: str = "m6", kp: float | None = None,
         max_current: float | None = XL330["max_current"], root: Path | None = None) -> Bam:
    root = root or Path(__file__).parent / "params"
    fitted = json.loads((root / motor / f"{model}.json").read_text())
    return Bam(
        kt=fitted["kt"],
        resistance=fitted["R"],
        armature=fitted["armature"],
        kp=XL330["kp"] if kp is None else kp,
        error_gain=XL330["error_gain"],
        max_pwm=XL330["max_pwm"],
        max_current=0.0 if max_current is None else max_current,
        friction_base=fitted.get("friction_base", 0.0),
        friction_viscous=fitted.get("friction_viscous", 0.0),
        friction_stribeck=fitted.get("friction_stribeck", 0.0),
        dtheta_stribeck=fitted.get("dtheta_stribeck", 1.0),
        alpha=fitted.get("alpha", 1.0),
        load_friction_base=fitted.get("load_friction_base", 0.0),
        load_friction_stribeck=fitted.get("load_friction_stribeck", 0.0),
        load_friction_motor=fitted.get("load_friction_motor", 0.0),
        load_friction_external=fitted.get("load_friction_external", 0.0),
        load_friction_motor_stribeck=fitted.get("load_friction_motor_stribeck", 0.0),
        load_friction_external_stribeck=fitted.get("load_friction_external_stribeck", 0.0),
        load_friction_motor_quad=fitted.get("load_friction_motor_quad", 0.0),
        load_friction_external_quad=fitted.get("load_friction_external_quad", 0.0),
        limited=max_current is not None,
        **FLAGS[model],
    )


def volts(bam: Bam, target, position, velocity, vin, kp):
    duty = (target - position) * kp * bam.error_gain

    if bam.limited:
        span = bam.resistance * bam.max_current / vin
        centre = bam.kt * velocity / vin
        duty = jnp.clip(duty, centre - span, centre + span)

    return vin * jnp.clip(duty, -bam.max_pwm, bam.max_pwm)


def torque(bam: Bam, volts, velocity):
    return bam.kt * volts / bam.resistance - bam.kt**2 * velocity / bam.resistance


def budget(bam: Bam, motor_torque, external_torque, stribeck):
    friction = jnp.full_like(motor_torque, bam.friction_base)

    if bam.stribeck:
        friction = friction + stribeck * bam.friction_stribeck

    if bam.load_dependent:
        if bam.directional:
            gearbox = jnp.abs(
                external_torque * bam.load_friction_external
                - motor_torque * bam.load_friction_motor
            )
            friction = friction + gearbox

            if bam.stribeck:
                gearbox_stribeck = jnp.abs(
                    external_torque * bam.load_friction_external_stribeck
                    - motor_torque * bam.load_friction_motor_stribeck
                )
                friction = friction + stribeck * gearbox_stribeck

                if bam.quadratic:
                    external = jnp.abs(external_torque)
                    motor = jnp.abs(motor_torque)
                    driving = (motor > external).astype(motor_torque.dtype)
                    quadratic = (
                        driving * bam.load_friction_external_quad * external**2
                        + (1.0 - driving) * bam.load_friction_motor_quad * motor**2
                    )
                    friction = friction + stribeck * quadratic
        else:
            gearbox = jnp.abs(external_torque - motor_torque)
            friction = friction + bam.load_friction_base * gearbox
            if bam.stribeck:
                friction = friction + stribeck * bam.load_friction_stribeck * gearbox

    return friction


@struct.dataclass
class Drive:
    torque: jax.Array
    frictionloss: jax.Array
    damping: jax.Array
    motor_torque: jax.Array


def drive(bam: Bam, target, position, velocity, external_torque,
          previous_actuator_torque, previous_motor_torque, vin,
          kp_scale=1.0, kd_scale=1.0, friction_scale=1.0,
          vin_drop_gain=0.0, vin_min=0.0):
    load = jnp.sum(jnp.abs(previous_motor_torque), axis=-1, keepdims=True)
    vin = jnp.maximum(vin - vin_drop_gain * load, vin_min)

    scaled = velocity * kd_scale
    control = volts(bam, target, position, scaled, vin, bam.kp * kp_scale)
    motor_torque = torque(bam, control, scaled)

    if bam.stribeck:
        stribeck = jnp.exp(-jnp.power(jnp.abs(velocity) / bam.dtheta_stribeck, bam.alpha))
    else:
        stribeck = jnp.zeros_like(velocity)

    frictionloss = budget(bam, previous_actuator_torque, external_torque, stribeck)

    return Drive(
        torque=motor_torque,
        frictionloss=frictionloss * friction_scale,
        damping=jnp.full_like(frictionloss, bam.friction_viscous),
        motor_torque=motor_torque,
    )
