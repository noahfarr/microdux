import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from microdux import actuator

GOLDEN = Path(__file__).parent / "golden_bam.json"


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="module")
def bam():
    return actuator.load(motor="xl330", model="m6", kp=200.0)


def arrays(golden):
    return {k: jnp.asarray(v, dtype=jnp.float64) for k, v in golden["inputs"].items()}


def test_fitted_parameters_are_the_upstream_ones(bam):
    assert bam.kt == pytest.approx(0.36601349688984386)
    assert bam.resistance == pytest.approx(2.8113923539223227)
    assert bam.armature == pytest.approx(0.0018077432831600838)
    assert (bam.load_dependent, bam.directional, bam.stribeck, bam.quadratic) == (
        True, True, True, True,
    )


def test_voltage_matches_upstream(bam, golden):
    x = arrays(golden)
    control = actuator.volts(
        bam, x["target"], x["pos"], x["vel"] * x["kd_scale"], x["vin"],
        bam.kp * x["kp_scale"],
    )
    np.testing.assert_allclose(control, np.asarray(golden["control"]), rtol=1e-12, atol=1e-12)


def test_torque_matches_upstream(bam, golden):
    x = arrays(golden)
    control = jnp.asarray(golden["control"], dtype=jnp.float64)
    got = actuator.torque(bam, control, x["vel"] * x["kd_scale"])
    np.testing.assert_allclose(got, np.asarray(golden["motor_torque"]), rtol=1e-12, atol=1e-12)


def test_friction_budget_matches_upstream(bam, golden):
    x = arrays(golden)
    stribeck = jnp.exp(-jnp.power(jnp.abs(x["vel"]) / bam.dtheta_stribeck, bam.alpha))
    got = actuator.budget(bam, x["prev"], x["ext"], stribeck) * x["friction_scale"]
    np.testing.assert_allclose(got, np.asarray(golden["frictionloss"]), rtol=1e-12, atol=1e-12)


def test_drive_composes_the_same_way(bam, golden):
    x = arrays(golden)
    out = actuator.drive(
        bam,
        target=x["target"],
        position=x["pos"],
        velocity=x["vel"],
        external_torque=x["ext"],
        previous_actuator_torque=x["prev"],
        previous_motor_torque=jnp.zeros_like(x["vel"]),
        vin=x["vin"],
        kp_scale=x["kp_scale"],
        kd_scale=x["kd_scale"],
        friction_scale=x["friction_scale"],
    )
    np.testing.assert_allclose(out.torque, np.asarray(golden["motor_torque"]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out.frictionloss, np.asarray(golden["frictionloss"]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out.damping, golden["viscous"], rtol=1e-12)


def test_voltage_sag_drops_the_supply_under_load(bam, golden):
    x = arrays(golden)
    heavy = jnp.full_like(x["vel"], 0.3)
    quiet = actuator.drive(
        bam, x["target"], x["pos"], x["vel"], x["ext"], x["prev"],
        jnp.zeros_like(x["vel"]), x["vin"], vin_drop_gain=0.2, vin_min=6.0,
    )
    loaded = actuator.drive(
        bam, x["target"], x["pos"], x["vel"], x["ext"], x["prev"],
        heavy, x["vin"], vin_drop_gain=0.2, vin_min=6.0,
    )
    assert jnp.abs(loaded.torque).sum() < jnp.abs(quiet.torque).sum()


def test_current_limit_can_be_disabled(golden):
    x = arrays(golden)
    free = actuator.load(motor="xl330", model="m6", kp=200.0, max_current=None)
    assert not free.limited
    unlimited = actuator.volts(free, x["target"], x["pos"], x["vel"], x["vin"], free.kp)
    limited = actuator.volts(
        actuator.load(motor="xl330", model="m6", kp=200.0),
        x["target"], x["pos"], x["vel"], x["vin"], 200.0,
    )
    assert not np.allclose(unlimited, limited)
