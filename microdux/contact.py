import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from mujoco.mjx._src import support

from . import backend


@struct.dataclass
class Gait:
    air_time: jax.Array
    contact_time: jax.Array
    last_air_time: jax.Array
    last_contact: jax.Array
    swing_peak: jax.Array


def pairs(data):
    found = backend.contacts(data)
    return found.geom1, found.geom2, found.dist


def member(ids, group):
    if isinstance(group, int):
        return ids == group
    mask = jnp.zeros_like(ids, dtype=bool)
    for g in group:
        mask = mask | (ids == g)
    return mask


def touching(data, geoms) -> jax.Array:
    geom1, geom2, dist = pairs(data)
    live = dist < 0
    hit = lambda g: jnp.any(live & (member(geom1, g) | member(geom2, g)))
    return jnp.array([hit(g) for g in geoms])


def forces(mjx_model, data, geoms, condim: int = 3, partner=None) -> jax.Array:
    contact = backend.contacts(data)
    geom1, geom2, dist = contact.geom1, contact.geom2, contact.dist
    rows = 2 * (condim - 1)

    address = contact.efc_address
    if address.ndim == 1:
        address = address[:, None] + np.arange(rows)[None]

    gathered = backend.efc_force(data)[address[:, :rows]]
    decoded = jax.vmap(support._decode_pyramid, in_axes=(0, 0, None))(
        gathered, contact.friction, condim
    )
    world = jax.vmap(lambda f, frame: (f[:3] @ frame))(decoded, contact.frame)

    live = (dist < 0) & (address[:, 0] >= 0)
    def net(g):
        if partner is None:
            mask = member(geom1, g) | member(geom2, g)
        else:
            mask = (member(geom1, g) & member(geom2, partner)) | (
                member(geom2, g) & member(geom1, partner))
        return jnp.sum(world * mask[:, None] * live[:, None], axis=0)

    return jnp.stack([net(g) for g in geoms])


def subtree(mj_model, body: str) -> tuple:
    import mujoco

    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body)
    ids = []
    for g in range(mj_model.ngeom):
        b = mj_model.geom_bodyid[g]
        while b != 0:
            if b == body_id:
                ids.append(g)
                break
            b = mj_model.body_parentid[b]
    return tuple(g for g in ids if mj_model.geom_contype[g] or mj_model.geom_conaffinity[g])


def rest(count: int) -> Gait:
    return Gait(
        air_time=jnp.zeros(count),
        contact_time=jnp.zeros(count),
        last_air_time=jnp.zeros(count),
        last_contact=jnp.zeros(count, dtype=bool),
        swing_peak=jnp.zeros(count),
    )


def tally(gait: Gait, contact: jax.Array, height: jax.Array, dt: float):
    landed = (gait.air_time > 0.0) & contact
    detached = (gait.contact_time > 0.0) & ~contact

    air_time = gait.air_time + dt
    contact_time = gait.contact_time + dt
    swing_peak = jnp.maximum(gait.swing_peak, height)

    updated = Gait(
        air_time=jnp.where(contact, 0.0, air_time),
        contact_time=jnp.where(contact, contact_time, 0.0),
        last_air_time=jnp.where(landed, air_time, gait.last_air_time),
        last_contact=contact,
        swing_peak=jnp.where(contact, 0.0, swing_peak),
    )
    return updated, detached, landed, jnp.where(contact, 0.0, air_time)
