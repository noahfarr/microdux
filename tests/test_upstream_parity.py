import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mujoco import mjx
from mujoco_playground._src import mjx_env

from microdux import commands, rewards, sense

GOLDEN = Path(__file__).parent / "golden_upstream.json"

BLOCKS = (
    ("angular velocity", slice(0, 3)),
    ("projected gravity", slice(3, 6)),
    ("joint position", slice(6, 20)),
    ("joint velocity", slice(20, 34)),
    ("last action", slice(34, 48)),
    ("commands", slice(48, 61)),
)


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip("upstream fixture not generated")
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

    info = dict(plain.reset(jax.random.key(0)).info)
    info["commands"] = commands.Commands(
        twist=jnp.asarray(golden["twist"]),
        head=jnp.asarray(golden["head"]),
        body=jnp.asarray(golden["body"]),
        twist_timer=info["commands"].twist_timer,
        head_timer=info["commands"].head_timer,
        body_timer=info["commands"].body_timer,
    )
    info["last_action"] = jnp.asarray(golden["last_action"])
    return env, data, info


def test_joint_order_matches_upstream(plain, golden):
    assert list(plain.env._layout.actuators) == list(golden["joint_names"])


def test_home_pose_matches_upstream_default(plain, golden):
    np.testing.assert_allclose(
        np.asarray(plain.env._home), golden["default_joint_pos"], atol=1e-5
    )


def test_observation_matches_upstream(posed, golden):
    env, data, info = posed
    ours = np.asarray(env._observe(data, info)["state"])
    theirs = np.asarray(golden["actor"])

    assert ours.shape == theirs.shape == (61,)
    for name, span in BLOCKS:
        np.testing.assert_allclose(
            ours[span], theirs[span], atol=1e-5, err_msg=f"{name} block diverged"
        )


def test_root_frames_match_upstream(posed, golden):
    _, data, _ = posed
    np.testing.assert_allclose(
        np.asarray(sense.root_linear_velocity(data)),
        golden["root_lin_vel_b"], atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(sense.root_angular_velocity(data)),
        golden["root_ang_vel_b"], atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(sense.gravity(data, sense.sensors(_env_model(posed)))),
        golden["projected_gravity_b"], atol=1e-5,
    )


def _env_model(posed):
    return posed[0].mj_model


def test_state_only_rewards_match_upstream(posed, golden):
    env, data, info = posed
    spans = env._sensors
    twist = info["commands"].twist
    joints = data.qpos[env._wiring.qpos_adr]
    tune = env._tuning

    ours = {
        "track_linear_velocity": rewards.track_linear_velocity(
            sense.root_linear_velocity(data), twist, tune.linear_std),
        "track_angular_velocity": rewards.track_angular_velocity(
            sense.root_angular_velocity(data), twist, tune.angular_std),
        "upright": rewards.upright(sense.tilt(data, spans), tune.upright_std),
        "pose": rewards.variable_posture(
            joints[env._leg_slots], env._home[env._leg_slots], twist,
            env._standing_std, env._walking_std, env._walking_std,
            tune.walking_threshold, tune.running_threshold),
        "body_ang_vel": rewards.body_angular_velocity(sense.world_angular_velocity(data)),
        "angular_momentum": rewards.angular_momentum(sense.read(data, spans.angmom)),
        "dof_pos_limits": rewards.joint_pos_limits(joints, env._limits),
        "head_pose_tracking": rewards.head_pose_tracking(
            joints[env._head_slots], info["commands"].head,
            env._home[env._head_slots], tune.head_std),
    }

    for name, value in ours.items():
        np.testing.assert_allclose(
            float(value), golden["rewards"][name], rtol=1e-4, atol=1e-7,
            err_msg=f"reward term {name} diverged from upstream",
        )


def test_reward_weights_match_upstream(golden):
    ours = vars(rewards.Weights())
    for name, weight in golden["weights"].items():
        if name in ours:
            assert ours[name] == pytest.approx(weight), name
