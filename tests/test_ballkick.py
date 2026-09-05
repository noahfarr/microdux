import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microdux.ballkick import BallKick


@pytest.fixture(scope="module")
def env():
    return BallKick(envs=2)


@pytest.fixture(scope="module")
def reset(env):
    return jax.jit(jax.vmap(env.reset))


@pytest.fixture(scope="module")
def step(env):
    return jax.jit(jax.vmap(env.step))


@pytest.fixture(scope="module")
def started(env, reset):
    return reset(jax.random.split(jax.random.key(0), 2))


def test_it_builds_on_the_ball_scene(env):
    assert "ball" in env.xml_path
    assert env.mj_model.nu == 14


def test_reset_produces_a_finite_state(started):
    assert np.isfinite(np.asarray(started.obs["state"])).all()
    assert np.isfinite(np.asarray(started.obs["privileged_state"])).all()
    assert float(started.reward.sum()) == 0.0
    assert float(started.done.sum()) == 0.0


def test_ball_spawns_in_front_of_the_kicking_foot(env, started):
    ball_xy = np.asarray(started.data.qpos[:, env._ball_qpos_adr:env._ball_qpos_adr + 2])
    assert (ball_xy[:, 0] > 0.0).all()


def test_step_and_all_reward_terms_are_finite(env, step, started):
    state = step(started, jnp.zeros((2, env.action_size)))
    assert np.isfinite(np.asarray(state.obs["state"])).all()
    assert np.isfinite(np.asarray(state.obs["privileged_state"])).all()
    assert np.isfinite(np.asarray(state.reward)).all()
    assert set(np.asarray(state.done).tolist()).issubset({0.0, 1.0})

    names = set(vars(env._weights))
    seen = {n[len("reward/"):] for n in state.metrics if n.startswith("reward/")}
    assert names == seen
    for name in names:
        assert np.isfinite(np.asarray(state.metrics[f"reward/{name}"])).all(), name
