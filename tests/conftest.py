import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from microdux import Velocity, delay, randomize


class Rig:
    def __init__(self, env):
        self.env = env
        self.reset = jax.jit(env.reset)
        self.step = jax.jit(env.step)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def roll(self, steps, seed=0, action=None):
        state = self.reset(jax.random.key(seed))
        action = jnp.zeros(self.env.action_size) if action is None else action
        seen = [state]
        for _ in range(steps):
            state = self.step(state, action)
            seen.append(state)
        return seen


@pytest.fixture(scope="session")
def plain():
    return Rig(Velocity())


@pytest.fixture(scope="session")
def randomised():
    return Rig(Velocity(spec=randomize.Spec(), noise=delay.Noise()))


@pytest.fixture(scope="session")
def aligned():
    return Rig(Velocity(spec=randomize.Spec(imu_angle=0.0)))
