import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mujoco import mjx
from mujoco_playground._src import mjx_env

from microdux import contact, rewards, sense

GOLDEN = Path(__file__).parent / "golden_gait.json"


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip("gait fixture not generated")
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="module")
def posed(plain, golden):
    env = plain.env
    data = mjx.forward(
        env.mjx_model,
        mjx_env.make_data(
            env.mj_model,
            qpos=jnp.asarray(golden["qpos"]),
            qvel=jnp.asarray(golden["qvel"]),
            impl=env._impl,
        ),
    )
    return env, data


def test_foot_contact_detection_matches_upstream(posed, golden):
    env, data = posed
    ours = np.asarray(contact.touching(data, env._feet))
    theirs = np.asarray(golden["found"]) > 0
    np.testing.assert_array_equal(ours, theirs)


def test_foot_site_heights_match_upstream(posed, golden):
    env, data = posed
    ours = np.asarray(data.site_xpos[env._foot_sites][:, 2])
    np.testing.assert_allclose(ours, golden["foot_site_z"], atol=1e-5)


def test_upstreams_ray_sensor_sits_below_the_site(golden):
    offset = np.asarray(golden["foot_heights"]) - np.asarray(golden["foot_site_z"])
    assert offset.max() < 0
    assert abs(offset[0] - offset[1]) < 1e-5
    assert abs(offset.mean()) == pytest.approx(0.00222, abs=2e-4)


def test_foot_velocities_match_upstream(posed, golden):
    env, data = posed
    ours = np.asarray(sense.foot_velocity(data, env._sensors))
    np.testing.assert_allclose(ours, golden["foot_vel_w"], atol=1e-4)


def test_self_collision_count_matches_upstream(posed, golden):
    env, data = posed
    ours = float(rewards.self_collision(contact.touching(data, env._self_collision)))
    assert ours == pytest.approx(float(np.sum(golden["self_collision_found"])), abs=1e-6)


def test_gait_rewards_match_upstream(posed, golden):
    env, data = posed
    tune = env._tuning
    twist = jnp.asarray(golden["twist"])

    air = jnp.asarray(golden["current_air_time"])
    landed = jnp.asarray(golden["first_contact"]) > 0
    touching = jnp.asarray(golden["found"]) > 0
    heights = jnp.asarray(golden["foot_heights"])
    velocity = jnp.asarray(golden["foot_vel_w"])

    ours = {
        "air_time": rewards.feet_air_time(
            air, twist, tune.air_time_min, tune.air_time_max, tune.command_threshold),
        "foot_clearance": rewards.feet_clearance(
            heights, velocity, twist, tune.target_height, tune.command_threshold),
        "foot_slip": rewards.feet_slip(
            velocity, touching, twist, tune.command_threshold),
    }

    for name, value in ours.items():
        np.testing.assert_allclose(
            float(value), golden["rewards"][name], rtol=1e-4, atol=1e-7,
            err_msg=f"gait reward {name} diverged from upstream",
        )
