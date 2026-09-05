import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microdux import RollerStandUp

ENVS = 2


@pytest.fixture(scope="module")
def env():
    return RollerStandUp(envs=ENVS)


@pytest.fixture(scope="module")
def reset(env):
    return jax.jit(jax.vmap(env.reset))


@pytest.fixture(scope="module")
def step(env):
    return jax.jit(jax.vmap(env.step))


@pytest.fixture(scope="module")
def started(reset):
    keys = jax.random.split(jax.random.key(0), ENVS)
    return reset(keys)


def test_it_builds_on_the_rollers_robot(env):
    assert "rollers" in env.xml_path
    assert env.mj_model.nu == 14


def test_reset_and_step_produce_finite_observations(env, step, started):
    state = started
    action = jnp.zeros((ENVS, env.action_size))
    for _ in range(3):
        state = step(state, action)
    assert np.isfinite(np.asarray(state.obs["state"])).all()
    assert np.isfinite(np.asarray(state.obs["privileged_state"])).all()
    assert np.isfinite(np.asarray(state.reward)).all()
    assert np.all(np.isin(np.asarray(state.done), [0.0, 1.0]))


def test_all_reward_terms_are_present_and_finite(env, step, started):
    action = jnp.zeros((ENVS, env.action_size))
    state = step(started, action)
    names = set(vars(env._weights))
    logged = {n[len("reward/"):] for n in state.metrics if n.startswith("reward/")}
    assert names == logged
    for name in names:
        assert np.isfinite(np.asarray(state.metrics[f"reward/{name}"])).all(), name


def test_ground_state_spawn_height_is_within_the_configured_ranges(env, started):
    height = np.asarray(started.data.qpos[:, 2])
    assert np.all(height >= 0.07) and np.all(height <= 0.15)
