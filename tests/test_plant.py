import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from microdux import actuator, constants, contact, model, plant


@pytest.fixture(scope="module")
def rig():
    mj, layout = model.build("walk")
    mjx_model = mjx.put_model(mj)
    return mj, layout, mjx_model, plant.wire(mj, layout), actuator.load(kp=200.0)


def standing(mj, layout, mjx_model):
    data = mjx.make_data(mj).replace(
        qpos=jnp.asarray(mj.key_qpos[layout.keyframes["STAND"]])
    )
    return mjx.forward(mjx_model, data)


def test_friction_rows_map_one_to_one_onto_actuated_dofs(rig):
    _, layout, _, wiring, _ = rig
    assert wiring.friction_rows.size == wiring.friction_dofs.size == 14
    np.testing.assert_array_equal(wiring.friction_dofs, layout.actuated_qvel_adr)


def test_analytic_row_layout_matches_a_real_constraint_solve(rig):
    _, _, mjx_model, wiring, _ = rig
    np.testing.assert_array_equal(wiring.friction_rows, plant.solved_rows(mjx_model))


def test_bam_reproduces_the_stiffness_of_the_servo_it_replaces(rig):
    _, _, _, _, bam = rig
    effective = bam.kp * bam.error_gain * 7.5 * bam.kt / bam.resistance

    spec = mujoco.MjSpec.from_file(str(constants.XMLS / "scene_walk.xml"))
    servo = spec.compile()
    assert servo.actuator_gainprm[0, 0] == pytest.approx(effective, rel=0.05)


def test_friction_written_each_substep_reaches_the_solver(rig):
    mj, layout, mjx_model, wiring, bam = rig
    data = standing(mj, layout, mjx_model)
    servos = plant.rest(mj.nu)
    target = jnp.asarray(layout.home)

    @jax.jit
    def roll(scale):
        def once(carry, _):
            state, servo = carry
            advanced = plant.advance(mjx_model, state, servo, bam, target, wiring, 4)
            return (advanced[0], advanced[1]), None

        servo = servos.replace(friction_scale=jnp.asarray([scale]))
        (state, _), _ = jax.lax.scan(once, (data, servo), (), 25)
        return state

    nominal = roll(1.0)
    sticky = roll(200.0)
    moved = np.abs(
        np.asarray(nominal.qpos[wiring.qpos_adr]) - np.asarray(sticky.qpos[wiring.qpos_adr])
    ).max()
    assert moved > 1e-4, f"friction scale changed joint travel by only {moved}"


def test_per_env_servo_parameters_give_different_dynamics(rig):
    mj, layout, mjx_model, wiring, bam = rig
    data = standing(mj, layout, mjx_model)
    target = jnp.asarray(layout.home)

    def roll(vin):
        servos = plant.rest(mj.nu).replace(vin=vin)

        def once(carry, _):
            state, servo = carry
            advanced = plant.advance(mjx_model, state, servo, bam, target, wiring, 4)
            return (advanced[0], advanced[1]), None

        (out, _), _ = jax.lax.scan(once, (data, servos), (), 25)
        return out.qvel

    batched = jax.jit(jax.vmap(roll))(jnp.asarray([[6.5], [8.2]]))
    assert np.abs(np.asarray(batched[0]) - np.asarray(batched[1])).max() > 1e-6


def test_contact_forces_balance_weight_at_rest(rig):
    mj, layout, mjx_model, wiring, bam = rig
    data = standing(mj, layout, mjx_model)
    servos = plant.rest(mj.nu)
    target = jnp.asarray(layout.home)
    run = jax.jit(lambda d, s: plant.advance(mjx_model, d, s, bam, target, wiring, 4)[:2])
    for _ in range(400):
        data, servos = run(data, servos)

    everything = tuple(range(mj.ngeom))
    vertical = float(np.asarray(contact.forces(mjx_model, data, everything))[:, 2].sum())
    weight = float(mj.body_mass.sum() * 9.81)
    # every contact is counted once per geom of its pair, hence twice
    assert vertical / weight == pytest.approx(2.0, abs=0.5)


def test_gait_accumulates_air_time_and_resets_on_contact():
    gait = contact.rest(2)
    airborne = jnp.asarray([False, False])
    for _ in range(5):
        gait, _, _, _ = contact.tally(gait, airborne, jnp.asarray([0.02, 0.03]), 0.02)
    np.testing.assert_allclose(gait.air_time, 0.1, rtol=1e-6)
    assert float(gait.swing_peak.max()) == pytest.approx(0.03)

    landed = jnp.asarray([True, False])
    gait, filtered, first, air = contact.tally(gait, landed, jnp.zeros(2), 0.02)
    assert bool(first[0]) and not bool(first[1])
    assert float(gait.air_time[0]) == 0.0
    assert float(gait.last_air_time[0]) == pytest.approx(0.12, rel=1e-6)
    assert float(gait.air_time[1]) == pytest.approx(0.12, rel=1e-6)
