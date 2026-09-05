import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from microdux import Velocity, model, plant

BACKLASH_VARIANTS = ("walk_backlash", "backlash", "rollers_backlash")
PLAIN_VARIANTS = ("walk", "allcollisions", "rollers")


@pytest.mark.parametrize("variant", BACKLASH_VARIANTS)
def test_backlash_variants_compile(variant):
    mj, layout = model.build(variant)
    assert mj.nu == 14
    assert len(layout.actuators) == 14
    assert model.friction_rows(mj) == 14


@pytest.mark.parametrize("variant", BACKLASH_VARIANTS)
def test_passive_backlash_joints_are_never_actuated(variant):
    mj, layout = model.build(variant)
    joint_names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mj.njnt)]
    backlash_joints = [n for n in joint_names if n and n.endswith("_backlash")]
    assert len(backlash_joints) == 14

    targeted = {
        mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, int(mj.actuator_trnid[i, 0]))
        for i in range(mj.nu)
    }
    assert set(backlash_joints).isdisjoint(targeted)
    assert all(not name.startswith("passive_") for name in targeted)


@pytest.mark.parametrize("variant", BACKLASH_VARIANTS)
def test_backlash_joints_carry_no_friction_row(variant):
    mj, layout = model.build(variant)
    assert bool((layout.backlash_mask == 1.0).all())
    np.testing.assert_array_equal(mj.dof_frictionloss[layout.backlash_qvel_adr], 0.0)
    np.testing.assert_array_equal(mj.dof_armature[layout.backlash_qvel_adr], 0.001)


@pytest.mark.parametrize("variant", PLAIN_VARIANTS)
def test_plain_variants_have_no_backlash(variant):
    mj, layout = model.build(variant)
    assert bool((layout.backlash_mask == 0.0).all())
    np.testing.assert_array_equal(layout.backlash_qpos_adr, layout.actuated_qpos_adr)
    np.testing.assert_array_equal(layout.backlash_qvel_adr, layout.actuated_qvel_adr)


def test_rollers_backlash_scene_is_not_the_plain_rollers_scene():
    assert model.SCENES["rollers_backlash"] != model.SCENES["rollers"]


def test_rollers_backlash_keyframes_agree_with_the_plain_rollers_pose():
    plain, plain_layout = model.build("rollers")
    lash, lash_layout = model.build("rollers_backlash")
    assert lash.nq == plain.nq + 14

    for name in ("STAND", "SIT", "FOLD", "INIT"):
        plain_key = mujoco.mj_name2id(plain, mujoco.mjtObj.mjOBJ_KEY, name)
        lash_key = mujoco.mj_name2id(lash, mujoco.mjtObj.mjOBJ_KEY, name)
        np.testing.assert_allclose(
            plain.key_ctrl[plain_key], lash.key_ctrl[lash_key], err_msg=name
        )
        np.testing.assert_allclose(
            plain.key_qpos[plain_key][plain_layout.actuated_qpos_adr],
            lash.key_qpos[lash_key][lash_layout.actuated_qpos_adr],
            err_msg=name,
        )
        np.testing.assert_allclose(
            lash.key_qpos[lash_key][lash_layout.backlash_qpos_adr], 0.0, err_msg=name
        )


def test_readout_sums_backlash_when_present():
    raw = jnp.array([1.0, 2.0, 3.0, 4.0])
    adr = np.array([0, 1])
    backlash_adr = np.array([2, 3])
    mask = np.array([1.0, 0.0])
    got = plant.readout(raw, adr, backlash_adr, mask)
    np.testing.assert_allclose(got, [4.0, 2.0])


def test_readout_is_identity_without_backlash():
    raw = jnp.array([1.0, 2.0, 3.0])
    adr = np.array([0, 2])
    got = plant.readout(raw, adr, None, None)
    np.testing.assert_allclose(got, [1.0, 3.0])


@pytest.mark.parametrize("variant", ("walk_backlash", "backlash"))
def test_backlash_env_steps_with_finite_state(variant):
    env = Velocity(variant=variant)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.key(0))
    action = jnp.zeros(env.action_size)
    for _ in range(10):
        state = step(state, action)
    assert state.obs["state"].shape == (61,)
    assert bool(jnp.isfinite(state.obs["state"]).all())
    assert bool(jnp.isfinite(state.data.qpos).all())
    assert bool(jnp.isfinite(state.data.qvel).all())
    np.testing.assert_array_equal(np.asarray(env._wiring.backlash_mask), 1.0)


def test_encoder_reads_through_backlash_in_the_observation():
    env = Velocity(variant="backlash")
    state = jax.jit(env.reset)(jax.random.key(0))
    data = state.data

    backlash_adr = env._wiring.backlash_qpos_adr
    bumped = data.qpos.at[backlash_adr[0]].set(0.01)
    bumped_data = data.replace(qpos=bumped)

    baseline = env._observe(data, state.info)
    perturbed = env._observe(bumped_data, state.info)

    JOINT_POS = slice(6, 20)
    delta = np.asarray(perturbed["state"][JOINT_POS]) - np.asarray(baseline["state"][JOINT_POS])
    np.testing.assert_allclose(delta[0], 0.01, atol=1e-9)
    np.testing.assert_allclose(delta[1:], 0.0, atol=1e-9)


def test_ground_truth_privileged_state_bypasses_backlash():
    env = Velocity(variant="backlash")
    state = jax.jit(env.reset)(jax.random.key(0))
    data = state.data

    backlash_adr = env._wiring.backlash_qpos_adr
    bumped = data.qpos.at[backlash_adr[0]].set(0.01)
    bumped_data = data.replace(qpos=bumped)

    baseline = env._observe(data, state.info)
    perturbed = env._observe(bumped_data, state.info)

    JOINT_POS = slice(6, 20)
    np.testing.assert_allclose(
        np.asarray(perturbed["privileged_state"][JOINT_POS]),
        np.asarray(baseline["privileged_state"][JOINT_POS]),
        atol=1e-12,
    )


def test_plant_position_feedback_reads_through_backlash():
    from mujoco import mjx

    from microdux import actuator

    mj, layout = model.build("backlash")
    mjx_model = mjx.put_model(mj)
    wiring = plant.wire(mj, layout)
    bam = actuator.load(kp=200.0)

    data = mjx.make_data(mj).replace(
        qpos=jnp.asarray(mj.key_qpos[layout.keyframes["STAND"]])
    )
    data = mjx.forward(mjx_model, data)
    servos = plant.rest(mj.nu)
    target = jnp.asarray(layout.home)

    _, plain_servos = plant.substep(mjx_model, data, servos, bam, target, wiring)

    bumped = data.qpos.at[wiring.backlash_qpos_adr[0]].set(0.01)
    bumped_data = data.replace(qpos=bumped)
    _, bumped_servos = plant.substep(mjx_model, bumped_data, servos, bam, target, wiring)

    assert not np.isclose(
        float(plain_servos.previous_motor_torque[0]),
        float(bumped_servos.previous_motor_torque[0]),
    ), "the PD control torque did not react to the backlash-joint's contribution"
