import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microdux.roulade import Roulade


@pytest.fixture(scope="module")
def env():
    return Roulade(envs=2)


@pytest.fixture(scope="module")
def started(env):
    keys = jax.random.split(jax.random.key(0), 2)
    return jax.vmap(env.reset)(keys)


@pytest.fixture(scope="module")
def step_fn(env):
    return jax.jit(jax.vmap(env.step))


def test_it_builds_on_the_allcollisions_robot(env):
    assert env.xml_path.endswith("scene.xml")
    assert env.mj_model.nu == 14


def test_reset_produces_a_finite_state(started):
    assert np.isfinite(np.asarray(started.obs["state"])).all()
    assert np.isfinite(np.asarray(started.obs["privileged_state"])).all()
    assert np.asarray(started.reward == 0.0).all()
    assert np.asarray(started.done == 0.0).all()
    assert np.isfinite(np.asarray(started.info["roulade_accum"])).all()
    assert np.isfinite(np.asarray(started.info["roulade_max"])).all()
    assert np.isfinite(np.asarray(started.info["roulade_paid"])).all()


def test_step_produces_finite_reward_and_obs(env, started, step_fn):
    action = jnp.zeros((2, env.action_size))
    state = started
    for _ in range(5):
        state = step_fn(state, action)
    assert np.isfinite(np.asarray(state.obs["state"])).all()
    assert np.isfinite(np.asarray(state.obs["privileged_state"])).all()
    assert np.isfinite(np.asarray(state.reward)).all()
    assert np.isin(np.asarray(state.done), [0.0, 1.0]).all()
    assert np.isfinite(np.asarray(state.info["roulade_accum"])).all()
    assert np.isfinite(np.asarray(state.info["roulade_max"])).all()
    assert np.isfinite(np.asarray(state.info["roulade_paid"])).all()
    assert state.info["roulade_head_latch"].dtype == jnp.bool_


def test_all_reward_terms_are_present_and_finite(env, started, step_fn):
    state = step_fn(started, jnp.zeros((2, env.action_size)))
    names = set(vars(env._weights))
    reported = {n[len("reward/"):] for n in state.metrics if n.startswith("reward/")}
    assert names == reported
    for name in names:
        assert np.isfinite(np.asarray(state.metrics[f"reward/{name}"])).all(), name


def test_midroll_spawn_seeds_a_nonzero_accumulator(env):
    keys = jax.random.split(jax.random.key(7), 64)
    _, _, _, _, spawn_angle, head_latch = jax.jit(jax.vmap(env._spawn))(keys)
    spawn_angle = np.asarray(spawn_angle)
    head_latch = np.asarray(head_latch)
    assert (spawn_angle > 0.0).any()
    assert (spawn_angle == 0.0).any()
    assert (head_latch == (spawn_angle > 0.0)).all()
