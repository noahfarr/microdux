import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Recovery:
    upright: jax.Array
    height: jax.Array
    fallen_s: jax.Array
    armed: jax.Array
    taxed: jax.Array
    down_s: jax.Array


def rest(cos_tilt, height) -> Recovery:
    return Recovery(
        upright=cos_tilt,
        height=height,
        fallen_s=jnp.zeros(()),
        armed=jnp.zeros((), dtype=bool),
        taxed=jnp.zeros((), dtype=bool),
        down_s=jnp.zeros(()),
    )


def tally(recovery: Recovery, cos_tilt, height, dt, fallen, down, recovered, min_fallen_s):
    upright_progress = cos_tilt - recovery.upright
    height_progress = height - recovery.height

    fallen_s = jnp.where(fallen, recovery.fallen_s + dt, 0.0)
    latched = recovery.armed | (fallen_s >= min_fallen_s)
    armed = latched & ~recovered
    success = latched & recovered
    taxed = (recovery.taxed | fallen) & ~recovered
    down_s = jnp.where(down, recovery.down_s + dt, 0.0)

    updated = Recovery(cos_tilt, height, fallen_s, armed, taxed, down_s)
    return updated, upright_progress, height_progress, taxed, success, down_s
