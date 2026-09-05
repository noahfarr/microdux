import jax
import jax.numpy as jnp
from flax import struct

TWIST_RESAMPLE = (3.0, 8.0)
POSE_RESAMPLE = (2.0, 5.0)


@struct.dataclass
class Ranges:
    lin_x: tuple = (-0.4, 0.4)
    lin_y: tuple = (-0.3, 0.3)
    ang_z: tuple = (-1.0, 1.0)
    head: tuple = ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))
    body: tuple = (
        (-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005),
        (-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05),
    )
    standing_fraction: float = 0.02
    turn_fraction: float = 0.15


@struct.dataclass
class Commands:
    twist: jax.Array
    head: jax.Array
    body: jax.Array
    twist_timer: jax.Array
    head_timer: jax.Array
    body_timer: jax.Array


def _uniform(key, spans):
    spans = jnp.asarray(spans, dtype=jnp.float32)
    return jax.random.uniform(key, (spans.shape[0],), minval=spans[:, 0], maxval=spans[:, 1])


def sample_twist(key, ranges: Ranges, standing_fraction=None):
    base, standing_key, turn_key, sign_key, magnitude_key = jax.random.split(key, 5)
    twist = _uniform(base, (ranges.lin_x, ranges.lin_y, ranges.ang_z))

    if standing_fraction is None:
        standing_fraction = ranges.standing_fraction
    standing = jax.random.uniform(standing_key) <= standing_fraction
    turning = jax.random.uniform(turn_key) < ranges.turn_fraction

    reach = max(abs(ranges.ang_z[0]), abs(ranges.ang_z[1]))
    sign = jnp.where(jax.random.uniform(sign_key) < 0.5, -1.0, 1.0)
    magnitude = jax.random.uniform(magnitude_key, minval=0.4 * reach, maxval=reach)
    spin = jnp.array([0.0, 0.0, 1.0]) * sign * magnitude

    twist = jnp.where(turning, spin, twist)
    return jnp.where(standing & ~turning, jnp.zeros(3), twist)


def steps(key, span, dt):
    seconds = jax.random.uniform(key, minval=span[0], maxval=span[1])
    return jnp.ceil(seconds / dt).astype(jnp.int32)


def rest(key, ranges: Ranges, dt: float, head_ranges=None, body_ranges=None) -> Commands:
    twist_key, head_key, body_key, t1, t2, t3 = jax.random.split(key, 6)
    return Commands(
        twist=sample_twist(twist_key, ranges),
        head=_uniform(head_key, head_ranges if head_ranges is not None else ranges.head),
        body=_uniform(body_key, body_ranges if body_ranges is not None else ranges.body),
        twist_timer=steps(t1, TWIST_RESAMPLE, dt),
        head_timer=steps(t2, POSE_RESAMPLE, dt),
        body_timer=steps(t3, POSE_RESAMPLE, dt),
    )


def refresh(commands: Commands, key, ranges: Ranges, dt: float,
            head_ranges=None, body_ranges=None, standing_fraction=None) -> Commands:
    twist_key, head_key, body_key, t1, t2, t3 = jax.random.split(key, 6)

    twist_due = commands.twist_timer <= 0
    head_due = commands.head_timer <= 0
    body_due = commands.body_timer <= 0

    head_spans = head_ranges if head_ranges is not None else ranges.head
    body_spans = body_ranges if body_ranges is not None else ranges.body

    return Commands(
        twist=jnp.where(
            twist_due, sample_twist(twist_key, ranges, standing_fraction), commands.twist),
        head=jnp.where(head_due, _uniform(head_key, head_spans), commands.head),
        body=jnp.where(body_due, _uniform(body_key, body_spans), commands.body),
        twist_timer=jnp.where(twist_due, steps(t1, TWIST_RESAMPLE, dt), commands.twist_timer - 1),
        head_timer=jnp.where(head_due, steps(t2, POSE_RESAMPLE, dt), commands.head_timer - 1),
        body_timer=jnp.where(body_due, steps(t3, POSE_RESAMPLE, dt), commands.body_timer - 1),
    )


def vector(commands: Commands) -> jax.Array:
    return jnp.concatenate([commands.twist, commands.head, commands.body])


@struct.dataclass
class Posture:
    flag: jax.Array
    blend: jax.Array
    timer: jax.Array


def sample_flag(key, sit_prob):
    return (jax.random.uniform(key) < sit_prob) * 1.0


def rest_posture(key, sit_prob, dwell, dt) -> Posture:
    flag_key, timer_key = jax.random.split(key)
    flag = sample_flag(flag_key, sit_prob)
    return Posture(flag=flag, blend=flag, timer=steps(timer_key, dwell, dt))


def refresh_posture(posture: Posture, key, sit_prob, dwell, dt, ramp_s) -> Posture:
    flag_key, timer_key = jax.random.split(key)
    due = posture.timer <= 0
    flag = jnp.where(due, sample_flag(flag_key, sit_prob), posture.flag)
    rate = dt / jnp.maximum(ramp_s, 1e-6)
    blend = posture.blend + jnp.clip(flag - posture.blend, -rate, rate)
    timer = jnp.where(due, steps(timer_key, dwell, dt), posture.timer - 1)
    return Posture(flag=flag, blend=blend, timer=timer)
