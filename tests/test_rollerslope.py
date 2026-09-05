import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microdux import RollerSlope
from microdux.env import PRESERVE

ENVS = 2


@pytest.fixture(scope="module")
def env():
    return RollerSlope(envs=ENVS)


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


def test_it_builds_on_the_rollers_robot_with_terrain(env):
    assert "rollers" in env.xml_path
    assert env.mj_model.nu == 14
    assert env.mj_model.nhfield >= 1


def test_reset_produces_a_finite_state_at_level_zero(started):
    assert np.isfinite(np.asarray(started.obs["state"])).all()
    assert np.isfinite(np.asarray(started.obs["privileged_state"])).all()
    assert np.all(np.asarray(started.info[PRESERVE].level) == 0)


def test_step_produces_finite_reward_and_obs(env, step, started):
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


def test_entry_rolls_without_slipping_at_spawn(env, started):
    root_vx = np.asarray(started.data.qvel[:, 0])
    wheel_vel = np.asarray(started.data.qvel[:, np.asarray(env._rig.wheels)])
    expected_omega = root_vx[:, None] / env._tuning.wheel_radius
    np.testing.assert_allclose(wheel_vel, expected_omega, atol=1e-4)
