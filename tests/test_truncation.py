import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mujoco_playground._src import wrapper

from microdux import Velocity
from microdux.train import harness

LENGTH = 12
SEATS = 4


def roll(build, full_reset):
    env = Velocity(envs=SEATS)
    wrapped = build(env, episode_length=LENGTH, action_repeat=1, full_reset=full_reset)
    reset = jax.jit(wrapped.reset)
    step = jax.jit(wrapped.step)
    state = reset(jax.random.split(jax.random.PRNGKey(0), SEATS))
    idle = jnp.zeros((SEATS, env.action_size))

    ticks = []
    for tick in range(1, 2 * LENGTH + 2):
        state = step(state, idle)
        ticks.append((
            tick,
            float(np.asarray(state.done)[0]),
            float(np.asarray(state.info["truncation"])[0]),
        ))
    return ticks


@pytest.fixture(scope="module")
def bare():
    return roll(wrapper.wrap_for_brax_training, True)


@pytest.fixture(scope="module")
def wrapped():
    return roll(harness, True)


def test_the_time_limit_fires(wrapped):
    assert [tick for tick, done, _ in wrapped if done > 0] == [LENGTH, 2 * LENGTH]


def test_truncation_marks_every_time_limit(wrapped):
    for tick, done, truncation in wrapped:
        assert truncation == done, f"tick {tick}: done {done} truncation {truncation}"


def test_the_bare_wrapper_loses_truncation(bare):
    assert [tick for tick, done, _ in bare if done > 0] == [LENGTH, 2 * LENGTH]
    assert not [tick for tick, _, truncation in bare if truncation > 0]


def test_a_fall_is_not_a_truncation():
    env = Velocity(envs=SEATS)
    wrapped = harness(env, episode_length=10_000, action_repeat=1, full_reset=True)
    reset = jax.jit(wrapped.reset)
    step = jax.jit(wrapped.step)
    state = reset(jax.random.split(jax.random.PRNGKey(0), SEATS))
    idle = jnp.zeros((SEATS, env.action_size))

    fell = False
    for _ in range(400):
        state = step(state, idle)
        if float(np.asarray(state.done)[0]) > 0:
            fell = True
            assert float(np.asarray(state.info["truncation"])[0]) == 0.0
            break
    assert fell, "the duck never fell, so the termination path went untested"
