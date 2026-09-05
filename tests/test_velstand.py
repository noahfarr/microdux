import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microdux import recovery, sense
from microdux.env import Velocity
from microdux.velstand import RecoveryTuning, Spawn, VelStand


@pytest.fixture(scope="module")
def env():
    return VelStand()


@pytest.fixture(scope="module")
def started(env):
    return jax.jit(env.reset)(jax.random.key(0))


def stepped(env, state, action=None, times=1):
    action = jnp.zeros(env.action_size) if action is None else action
    step = jax.jit(env.step)
    for _ in range(times):
        state = step(state, action)
    return state


def test_it_builds_on_the_allcollisions_robot(env):
    assert "allcollisions" in env.xml_path
    assert env.mj_model.nu == 14


def test_reset_produces_a_finite_state(started):
    assert np.isfinite(np.asarray(started.obs["state"])).all()
    assert np.isfinite(np.asarray(started.obs["privileged_state"])).all()
    assert float(started.reward) == 0.0
    assert float(started.done) == 0.0


def test_step_produces_finite_reward_and_obs(env, started):
    state = stepped(env, started, times=3)
    assert np.isfinite(np.asarray(state.obs["state"])).all()
    assert np.isfinite(float(state.reward))
    assert float(state.done) in (0.0, 1.0)


def test_all_reward_terms_are_present_and_finite(env, started):
    state = stepped(env, started)
    names = env._names()
    assert set(names) == {n[len("reward/"):] for n in state.metrics if n.startswith("reward/")}
    for name in names:
        assert np.isfinite(float(state.metrics[f"reward/{name}"])), name


def test_default_spawn_matches_plain_velocity_upright_reset():
    stand = VelStand(spawn=Spawn())
    walk = Velocity(variant="allcollisions")
    a = jax.jit(stand.reset)(jax.random.key(3))
    b = jax.jit(walk.reset)(jax.random.key(3))
    np.testing.assert_allclose(np.asarray(a.data.qpos), np.asarray(b.data.qpos), atol=1e-6)


def test_prone_spawn_lies_the_robot_low_and_on_its_side():
    stand = VelStand(spawn=Spawn(prone_prob=1.0, face_down_prob=1.0))
    state = jax.jit(stand.reset)(jax.random.key(1))
    height = float(state.data.qpos[2])
    assert 0.05 <= height <= 0.09
    cos_tilt = float(sense.cos_tilt(state.data, stand._walk._sensors))
    assert abs(cos_tilt) < 0.2


def test_crouch_spawn_bends_the_legs_and_lowers_the_trunk():
    stand = VelStand(spawn=Spawn(prone_prob=0.0, crouch_prob=1.0))
    state = jax.jit(stand.reset)(jax.random.key(2))
    height = float(state.data.qpos[2])
    assert 0.05 <= height <= 0.13
    joints = np.asarray(state.data.qpos[stand._walk._wiring.qpos_adr])
    home = np.asarray(stand._walk._home)
    deviation = np.abs(joints[np.asarray(stand._crouch_slots)] - home[np.asarray(stand._crouch_slots)])
    assert deviation.max() > 0.1


def test_prone_reset_taxes_the_fallen_state_and_gates_air_time(env):
    stand = VelStand(spawn=Spawn(prone_prob=1.0, face_down_prob=1.0))
    state = jax.jit(stand.reset)(jax.random.key(4))
    state = stepped(stand, state)
    assert float(state.metrics["reward/fallen_tax"]) == 1.0
    assert float(state.metrics["reward/air_time"]) == 0.0
    assert float(state.metrics["reward/recovery_success"]) == 0.0
    assert np.isfinite(float(state.metrics["reward/upright_progress"]))
    assert np.isfinite(float(state.metrics["reward/height_progress"]))
    assert np.isfinite(float(state.metrics["reward/joint_torque_rate_l2"]))


def test_upright_walking_reset_is_not_taxed(env, started):
    state = stepped(env, started, times=2)
    assert float(state.metrics["reward/fallen_tax"]) == 0.0


def test_fallen_too_long_backstop_terminates_a_stuck_recovery():
    tune = RecoveryTuning(fallen_timeout_s=0.03)
    stand = VelStand(spawn=Spawn(prone_prob=1.0, face_down_prob=1.0), recovery_tuning=tune)
    state = jax.jit(stand.reset)(jax.random.key(5))
    state = stepped(stand, state, times=2)
    assert float(state.done) == 1.0


def test_fell_over_termination_disables_past_the_curriculum_threshold():
    huge = VelStand(spawn=Spawn(prone_prob=1.0, face_down_prob=1.0), envs=50_000_000)
    state = jax.jit(huge.reset)(jax.random.key(6))
    step = jax.jit(huge.step)
    action = jnp.zeros(huge.action_size)

    state = step(state, action)
    assert float(state.done) == 1.0

    state = step(state, action)
    assert float(state.done) == 0.0


def test_fell_over_termination_stays_active_without_the_curriculum():
    small = VelStand(spawn=Spawn(prone_prob=1.0, face_down_prob=1.0))
    state = jax.jit(small.reset)(jax.random.key(6))
    step = jax.jit(small.step)
    action = jnp.zeros(small.action_size)

    state = step(state, action)
    assert float(state.done) == 1.0

    state = step(state, action)
    assert float(state.done) == 1.0


def test_recovery_rest_seeds_the_potentials():
    state = recovery.rest(jnp.array(0.3), jnp.array(0.05))
    assert float(state.upright) == 0.3
    assert float(state.height) == 0.05
    assert not bool(state.armed)
    assert not bool(state.taxed)


def test_recovery_tally_arms_on_a_sustained_fall_and_fires_success_once():
    state = recovery.rest(jnp.array(0.0), jnp.array(0.05))
    for _ in range(5):
        state, upright_progress, height_progress, taxed, success, down_s = recovery.tally(
            state, jnp.array(0.0), jnp.array(0.05), 0.1,
            fallen=jnp.array(True), down=jnp.array(True), recovered=jnp.array(False),
            min_fallen_s=0.3,
        )
        assert not bool(success)
    assert bool(state.armed)
    assert bool(taxed)

    state, upright_progress, height_progress, taxed, success, down_s = recovery.tally(
        state, jnp.array(1.0), jnp.array(0.12), 0.1,
        fallen=jnp.array(False), down=jnp.array(False), recovered=jnp.array(True),
        min_fallen_s=0.3,
    )
    assert bool(success)
    assert not bool(state.armed)
    assert not bool(taxed)
    assert float(upright_progress) == pytest.approx(1.0)
    assert float(height_progress) == pytest.approx(0.07)


def test_recovery_tally_never_fires_success_without_a_real_fall():
    state = recovery.rest(jnp.array(0.9), jnp.array(0.11))
    state, _, _, taxed, success, _ = recovery.tally(
        state, jnp.array(0.95), jnp.array(0.115), 0.1,
        fallen=jnp.array(False), down=jnp.array(False), recovered=jnp.array(True),
        min_fallen_s=0.3,
    )
    assert not bool(success)
    assert not bool(taxed)
