import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mujoco_playground import wrapper

from microdux.env import PRESERVE

ENVS = 4


@functools.lru_cache(maxsize=None)
def harness(env, episode_length, full_reset):
    outer = wrapper.wrap_for_brax_training(
        env, episode_length=episode_length, full_reset=full_reset
    )
    return outer, jax.jit(outer.reset), jax.jit(outer.step)


def run(rig, steps, episode_length=8, full_reset=True):
    outer, reset, step = harness(rig.env, episode_length, full_reset)
    state = reset(jax.random.split(jax.random.PRNGKey(0), ENVS))
    seen = []
    for _ in range(steps):
        state = step(state, jnp.zeros((ENVS, rig.env.action_size)))
        seen.append(state)
    return seen


def test_randomisation_is_redrawn_on_every_episode(randomised):
    seen = run(randomised, steps=12)
    voltages = np.stack([np.asarray(s.info["servos"].vin).ravel() for s in seen])
    assert len(np.unique(voltages.round(6), axis=0)) > 1, (
        "battery voltage never changed, so randomisation is per run, not per episode"
    )


def test_without_full_reset_the_draw_is_frozen(randomised):
    seen = run(randomised, steps=12, full_reset=False)
    voltages = np.stack([np.asarray(s.info["servos"].vin).ravel() for s in seen])
    assert len(np.unique(voltages.round(6), axis=0)) == 1


def test_curriculum_counter_survives_episode_resets(randomised):
    seen = run(randomised, steps=12)
    counters = [int(np.asarray(s.info[PRESERVE]).ravel()[0]) for s in seen]
    assert counters == list(range(1, 13)), counters


def test_episode_wrapper_owns_the_timeout(randomised):
    seen = run(randomised, steps=10, episode_length=4)
    steps = [int(np.asarray(s.info["steps"]).ravel()[0]) for s in seen]
    assert max(steps) <= 4
    assert steps == [1, 2, 3, 4, 1, 2, 3, 4, 1, 2], steps


def test_the_env_does_not_terminate_on_its_own_clock(plain):
    state = plain.roll(20)[-1]
    assert "episode" not in state.info
