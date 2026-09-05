import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microdux import Velocity, delay, randomize


@pytest.fixture(scope="module")
def env():
    return Velocity(spec=randomize.Spec(), noise=delay.Noise())


@pytest.fixture(scope="module")
def batch(env):
    keys = jax.random.split(jax.random.key(0), 8)
    return jax.jit(jax.vmap(env.reset))(keys)


def spread(values):
    values = np.asarray(values)
    return float(values.max() - values.min())


def test_every_randomised_field_varies_across_envs(batch):
    drawn = batch.info["draw"]
    assert spread(batch.info["servos"].vin) > 0
    assert spread(batch.info["servos"].vin_drop_gain) > 0
    assert spread(batch.info["servos"].friction_scale) > 0
    assert spread(drawn.body_mass[:, 1]) > 0
    assert spread(drawn.body_inertia[:, 1, 0]) > 0
    assert spread(drawn.dof_armature[:, 6]) > 0
    assert spread(drawn.trunk_com[:, 0, 0]) > 0
    assert spread(drawn.head_com[:, 0, 0]) > 0
    assert spread(drawn.encoder_bias[:, 0]) > 0


def test_ranges_match_upstream(env, batch):
    spec = randomize.Spec()
    drawn = batch.info["draw"]
    servos = batch.info["servos"]

    assert np.asarray(servos.vin).min() >= spec.vin[0]
    assert np.asarray(servos.vin).max() <= spec.vin[1]
    assert np.abs(np.asarray(drawn.encoder_bias)).max() <= spec.encoder_bias[1]

    mu = np.asarray(drawn.geom_friction[:, env._feet[0], 0])
    assert mu.min() >= spec.foot_friction[0] and mu.max() <= spec.foot_friction[1]

    nominal = np.asarray(env.mjx_model.dof_armature[6])
    scaled = np.asarray(drawn.dof_armature[:, 6]) / nominal
    assert scaled.min() >= spec.armature[0] - 1e-6
    assert scaled.max() <= spec.armature[1] + 1e-6


def test_mass_and_inertia_scale_together(batch):
    drawn = batch.info["draw"]
    mass = np.asarray(drawn.body_mass[:, 1])
    inertia = np.asarray(drawn.body_inertia[:, 1, 0])
    np.testing.assert_allclose(mass / mass[0], inertia / inertia[0], rtol=1e-6)


def test_randomisation_reaches_the_dynamics(env):
    heavy = randomize.Spec(mass_inertia=(2.0, 2.0), com=0.0, head_com=0.0,
                           armature=(1.0, 1.0), joint_friction=(1.0, 1.0),
                           foot_friction=(1.0, 1.0), vin=(7.5, 7.5),
                           vin_drop_gain=(0.0, 0.0), encoder_bias=(0.0, 0.0),
                           imu_angle=0.0, base_pitch=0.0, base_roll=0.0,
                           base_height=(0.12, 0.12), push=(0.0, 0.0))
    light = heavy.replace(mass_inertia=(0.5, 0.5))

    def roll(spec):
        world = Velocity(spec=spec)
        state = jax.jit(world.reset)(jax.random.key(0))
        step = jax.jit(world.step)
        for _ in range(20):
            state = step(state, jnp.zeros(world.action_size))
        return np.asarray(state.data.qpos)

    assert np.abs(roll(heavy) - roll(light)).max() > 1e-3


def test_observation_noise_is_off_without_a_noise_spec():
    quiet = Velocity(spec=randomize.Spec(imu_angle=0.0, encoder_bias=(0.0, 0.0)))
    state = jax.jit(quiet.reset)(jax.random.key(0))
    step = jax.jit(quiet.step)
    a = step(state, jnp.zeros(quiet.action_size)).obs["state"]
    b = step(state, jnp.zeros(quiet.action_size)).obs["state"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b))


def test_pushes_perturb_the_base(env):
    always = randomize.Spec(push_interval=(0.0, 0.0), push=(1.0, 1.0))
    world = Velocity(spec=always)
    state = jax.jit(world.reset)(jax.random.key(0))
    before = float(jnp.linalg.norm(state.data.qvel[:2]))
    after = jax.jit(world.step)(state, jnp.zeros(world.action_size))
    assert float(jnp.linalg.norm(after.data.qvel[:2])) > before


def test_env_without_randomisation_keeps_the_nominal_model():
    plain = Velocity()
    state = jax.jit(plain.reset)(jax.random.key(0))
    assert state.info["draw"] is None
    np.testing.assert_allclose(float(state.data.qpos[2]), 0.12, atol=1e-6)
