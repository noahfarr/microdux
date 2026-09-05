import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microdux.rollers import Rollers


@pytest.fixture(scope="module")
def env():
    return Rollers(envs=2)


@pytest.fixture(scope="module")
def stepper(env):
    return jax.jit(jax.vmap(env.step))


@pytest.fixture(scope="module")
def started(env):
    keys = jax.random.split(jax.random.key(0), 2)
    return jax.jit(jax.vmap(env.reset))(keys)


def stepped(env, stepper, state, times=1):
    action = jnp.zeros((2, env.action_size))
    for _ in range(times):
        state = stepper(state, action)
    return state


def test_it_builds_on_the_rollers_robot(env):
    assert "rollers" in env.xml_path
    assert env.mj_model.nu == 14


def test_reset_produces_a_finite_state(started):
    assert np.isfinite(np.asarray(started.obs["state"])).all()
    assert np.isfinite(np.asarray(started.obs["privileged_state"])).all()
    assert np.all(np.asarray(started.reward) == 0.0)
    assert np.all(np.asarray(started.done) == 0.0)


def test_observation_and_action_shapes(env, started):
    assert env.action_size == 14
    assert started.obs["state"].shape == (2, 61)


def test_step_produces_finite_reward_obs_and_all_terms(env, stepper, started):
    state = stepped(env, stepper, started, times=5)
    assert np.isfinite(np.asarray(state.obs["state"])).all()
    assert np.isfinite(np.asarray(state.obs["privileged_state"])).all()
    assert np.isfinite(np.asarray(state.reward)).all()
    assert set(np.unique(np.asarray(state.done))) <= {0.0, 1.0}

    names = set(vars(env._weights))
    assert names == {n[len("reward/"):] for n in state.metrics if n.startswith("reward/")}
    for name in names:
        assert np.isfinite(np.asarray(state.metrics[f"reward/{name}"])).all(), name

    assert np.all(np.asarray(state.info["swing_accum"]) >= 0.0)
