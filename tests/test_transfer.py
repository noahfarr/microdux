from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microdux import Velocity, delay, render

WEIGHTS = Path(__file__).parent / "fixtures" / "upstream_policy.npz"
STEPS = 250


@pytest.fixture(scope="module")
def upstream():
    if not WEIGHTS.exists():
        pytest.skip("upstream policy fixture not present")
    blob = np.load(WEIGHTS)
    mean = jnp.asarray(blob["obs_normalizer._mean"][0])
    std = jnp.asarray(blob["obs_normalizer._std"][0])
    layers = [
        (jnp.asarray(blob[f"mlp.{i}.weight"]), jnp.asarray(blob[f"mlp.{i}.bias"]))
        for i in (0, 2, 4, 6)
    ]

    def act(obs, key):
        x = (obs["state"] - mean) / jnp.maximum(std, 1e-8)
        for index, (weight, bias) in enumerate(layers):
            x = x @ weight.T + bias
            if index < len(layers) - 1:
                x = jax.nn.elu(x)
        return x

    return act


def test_upstream_policy_stays_upright_here(plain, upstream):
    qpos, rewards, dones = render.rollout(plain.env, policy=upstream, steps=STEPS)
    assert len(qpos) == STEPS, (
        f"upstream's trained policy fell after {len(qpos)} steps; our dynamics "
        "have diverged from the environment it was trained in"
    )
    assert float(qpos[-1][2]) > 0.08


def test_upstream_policy_makes_progress_here(plain, upstream):
    qpos, _, _ = render.rollout(plain.env, policy=upstream, steps=STEPS)
    travelled = float(np.linalg.norm(qpos[-1][:2] - qpos[0][:2]))
    assert travelled > 0.05, f"upstream's policy only moved {travelled:.3f} m"


def test_actuator_delay_is_counted_in_substeps(plain):
    assert plain.env._substep_lag.min_lag == 3
    assert plain.env._substep_lag.max_lag == 6
    assert plain.env.n_substeps == 4
    held = plain.env._stand
    assert plain.roll(1)[-1].info["targets"].shape == (
        delay.ACTION.max_lag + 1,
        plain.env.action_size,
    )
