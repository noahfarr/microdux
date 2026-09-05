import jax
import jax.numpy as jnp
import pytest
from brax.envs.wrappers import training as brax_wrapper
from mujoco_playground import wrapper

from microdux import Velocity, wrappers

ENVS = 4
STEPS = 12
EPISODE = 5


def rollout(harnessed):
    state = jax.jit(harnessed.reset)(jax.random.split(jax.random.PRNGKey(0), ENVS))
    action = jnp.zeros((ENVS, harnessed.action_size))
    step = jax.jit(harnessed.step)
    seen = []
    for _ in range(STEPS):
        state = step(state, action)
        seen.append((state.data.qpos, state.info["truncation"], state.done))
    return seen


def brax(env):
    env = brax_wrapper.VmapWrapper(env)
    env = brax_wrapper.EpisodeWrapper(env, EPISODE, 1)
    env = wrapper.BraxAutoResetWrapper(env, full_reset=True)
    return wrappers.Bootstrapping(env, EPISODE)


def ours(env):
    env = brax_wrapper.VmapWrapper(env)
    env = brax_wrapper.EpisodeWrapper(env, EPISODE, 1)
    env = wrappers.AutoReset(env)
    return wrappers.Bootstrapping(env, EPISODE)


@pytest.fixture(scope="module")
def env():
    return Velocity(envs=ENVS)


def test_autoreset_matches_brax_without_a_respawn_hook(env):
    theirs = rollout(brax(env))
    mine = rollout(ours(env))
    for (qpos, trunc, done), (mqpos, mtrunc, mdone) in zip(theirs, mine):
        assert jnp.allclose(qpos, mqpos, atol=1e-6)
        assert jnp.array_equal(trunc, mtrunc)
        assert jnp.array_equal(done, mdone)


def test_autoreset_carries_the_counter_across_episodes(env):
    harnessed = ours(env)
    state = jax.jit(harnessed.reset)(jax.random.split(jax.random.PRNGKey(0), ENVS))
    action = jnp.zeros((ENVS, harnessed.action_size))
    step = jax.jit(harnessed.step)
    for _ in range(STEPS):
        state = step(state, action)
    assert int(jnp.max(state.info[wrappers.PRESERVE])) >= EPISODE


def test_respawn_moves_the_ground_mix_with_the_counter():
    import microdux
    from microdux import curricula

    task = microdux.StandUp(envs=4)
    keys = jax.random.split(jax.random.PRNGKey(0), 32)

    def heights(preserved):
        weights = curricula.ranges(jnp.int32(preserved), curricula.GROUND_MIX)
        mix = task._ground_mix.replace(
            standing=weights[0], sitting=weights[1],
            face_down=weights[2], face_up=weights[3],
        )
        spawn = jax.jit(jax.vmap(lambda k: task.reset(k, mix).data.qpos[2]))
        return spawn(keys)

    early = heights(0)
    late = heights(curricula.iterations(2500))
    assert not jnp.allclose(early, late)


def test_ground_mix_matches_upstream_stages():
    from microdux import curricula

    stages = {
        0: (0.40, 0.40, 0.20, 0.00),
        600: (0.25, 0.30, 0.35, 0.10),
        1500: (0.20, 0.25, 0.30, 0.25),
        2500: (0.15, 0.20, 0.30, 0.35),
    }
    for iteration, expected in stages.items():
        got = curricula.ranges(
            jnp.int32(curricula.iterations(iteration)), curricula.GROUND_MIX
        )
        assert jnp.allclose(got, jnp.array(expected)), (iteration, got)
