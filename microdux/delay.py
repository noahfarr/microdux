import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Line:
    min_lag: int = struct.field(pytree_node=False)
    max_lag: int = struct.field(pytree_node=False)
    update_period: int = struct.field(pytree_node=False)
    width: int = struct.field(pytree_node=False)


@struct.dataclass
class Buffer:
    history: jax.Array
    lag: jax.Array
    phase: jax.Array


def rest(line: Line, value) -> Buffer:
    return Buffer(
        history=jnp.tile(value, (line.max_lag + 1, 1)),
        lag=jnp.asarray(line.min_lag, jnp.int32),
        phase=jnp.zeros((), jnp.int32),
    )


def start(line: Line, value, key) -> Buffer:
    lag_key, phase_key = jax.random.split(key)
    phase = jnp.where(
        line.update_period > 0,
        jax.random.randint(phase_key, (), 0, max(line.update_period, 1)),
        jnp.zeros((), jnp.int32),
    )
    return Buffer(
        history=jnp.tile(value, (line.max_lag + 1, 1)),
        lag=jax.random.randint(lag_key, (), line.min_lag, line.max_lag + 1),
        phase=phase.astype(jnp.int32),
    )


def push(buffer: Buffer, line: Line, value, key, step) -> tuple[Buffer, jax.Array]:
    history = jnp.roll(buffer.history, 1, axis=0).at[0].set(value)

    if line.max_lag > line.min_lag:
        due = jnp.where(
            line.update_period > 0,
            jnp.mod(step + buffer.phase, max(line.update_period, 1)) == 0,
            jnp.asarray(True),
        )
        fresh = jax.random.randint(key, (), line.min_lag, line.max_lag + 1)
        lag = jnp.where(due, fresh, buffer.lag)
    else:
        lag = buffer.lag

    return Buffer(history=history, lag=lag, phase=buffer.phase), history[lag]


IMU = Line(min_lag=0, max_lag=1, update_period=64, width=3)
GRAVITY = Line(min_lag=0, max_lag=1, update_period=64, width=3)
JOINT_VEL = Line(min_lag=1, max_lag=1, update_period=0, width=14)
ACTION = Line(min_lag=3, max_lag=6, update_period=0, width=14)


@struct.dataclass
class Noise:
    ang_vel: float = 0.03
    gravity: float = 0.01
    joint_pos: float = 0.001
    joint_vel: float = 0.25


def jitter(key, value, amplitude):
    return value + jax.random.uniform(
        key, value.shape, minval=-amplitude, maxval=amplitude
    )
