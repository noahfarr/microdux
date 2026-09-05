import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microdux import contact

ANG_VEL = slice(0, 3)
GRAVITY = slice(3, 6)
JOINT_POS = slice(6, 20)
JOINT_VEL = slice(20, 34)
LAST_ACTION = slice(34, 48)
TWIST = slice(48, 51)
HEAD = slice(51, 55)
BODY = slice(55, 61)


def test_observation_layout_matches_upstream(plain):
    state = plain.roll(0)[0]
    assert state.obs["state"].shape == (61,)
    obs = np.asarray(state.obs["state"])
    np.testing.assert_allclose(obs[LAST_ACTION], 0.0)
    np.testing.assert_allclose(
        obs[TWIST], np.asarray(state.info["commands"].twist), rtol=1e-6
    )
    np.testing.assert_allclose(
        obs[HEAD], np.asarray(state.info["commands"].head), rtol=1e-6
    )
    np.testing.assert_allclose(
        obs[BODY], np.asarray(state.info["commands"].body), rtol=1e-6
    )


def test_joint_velocity_observation_lags_one_control_step(plain):
    seen = plain.roll(5)
    adr = plain.env._wiring.qvel_adr
    for previous, current in zip(seen[1:-1], seen[2:]):
        np.testing.assert_allclose(
            np.asarray(current.obs["state"][JOINT_VEL]),
            np.asarray(previous.data.qvel[adr]),
            atol=1e-6,
        )


def test_joint_position_is_not_delayed(plain):
    state = plain.roll(1)[-1]
    live = np.asarray(state.data.qpos[plain.env._wiring.qpos_adr]) - np.asarray(plain.env._home)
    np.testing.assert_allclose(np.asarray(state.obs["state"][JOINT_POS]), live, atol=1e-6)


def test_imu_terms_are_delayed_when_randomised(aligned):
    for state in aligned.roll(6)[1:]:
        buffer = state.info["imu"]
        np.testing.assert_allclose(
            np.asarray(state.obs["state"][ANG_VEL]),
            np.asarray(buffer.history[int(buffer.lag)]),
            atol=1e-6,
        )


def test_air_time_matches_upstream_bookkeeping():
    gait = contact.rest(2)
    grounded = jnp.asarray([True, True])
    airborne = jnp.asarray([False, False])

    gait, _, _, air = contact.tally(gait, grounded, jnp.zeros(2), 0.02)
    np.testing.assert_allclose(air, 0.0)
    np.testing.assert_allclose(gait.contact_time, 0.02)

    for expected in (0.02, 0.04, 0.06):
        gait, detached, landed, air = contact.tally(gait, airborne, jnp.zeros(2), 0.02)
        np.testing.assert_allclose(air, expected)
    assert not bool(landed.any())

    gait, detached, landed, air = contact.tally(gait, grounded, jnp.zeros(2), 0.02)
    assert bool(landed.all())
    np.testing.assert_allclose(gait.last_air_time, 0.08)
    np.testing.assert_allclose(air, 0.0)


def test_liftoff_starts_counting_immediately():
    gait = contact.rest(2)
    gait, _, _, _ = contact.tally(gait, jnp.asarray([True, True]), jnp.zeros(2), 0.02)
    gait, detached, landed, air = contact.tally(
        gait, jnp.asarray([False, False]), jnp.zeros(2), 0.02
    )
    assert bool(detached.all())
    assert not bool(landed.any())
    np.testing.assert_allclose(air, 0.02)


def test_landing_is_not_reported_on_liftoff():
    gait = contact.rest(2)
    for contacts in ([True, True], [False, False], [False, False]):
        gait, detached, landed, air = contact.tally(
            gait, jnp.asarray(contacts), jnp.zeros(2), 0.02
        )
        assert not bool(landed.any()), contacts


def test_the_critic_sees_through_the_noise():
    import jax
    import numpy as np

    from microdux import Velocity, delay, randomize, sense

    env = Velocity(spec=randomize.Spec(), noise=delay.Noise())
    state = jax.jit(env.reset)(jax.random.key(0))
    key = jax.random.key(1)
    for _ in range(6):
        key, act_key = jax.random.split(key)
        state = jax.jit(env.step)(state, jax.random.normal(act_key, (env.action_size,)) * 0.1)

    actor = np.asarray(state.obs["state"])
    critic = np.asarray(state.obs["privileged_state"])
    width = actor.shape[-1]

    assert not np.allclose(actor, critic[:width])
    assert np.allclose(critic[:3], np.asarray(sense.read(state.data, env._sensors.ang_vel)))
    assert np.allclose(critic[3:6], np.asarray(sense.gravity(state.data, env._sensors)))
    joints = np.asarray(state.data.qpos[env._wiring.qpos_adr] - env._home)
    assert np.allclose(critic[6:20], joints)
