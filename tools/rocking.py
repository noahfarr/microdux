import sys
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

from microdux import Velocity, delay, randomize, sense
from microdux.train import networks

FIXTURE = "/home/farr/microdux/tests/fixtures/upstream_policy.npz"
blob = np.load(FIXTURE)
mean = jnp.asarray(blob["obs_normalizer._mean"][0])
scale = jnp.asarray(blob["obs_normalizer._std"][0])
layers = [
    (jnp.asarray(blob[f"mlp.{i}.weight"]), jnp.asarray(blob[f"mlp.{i}.bias"]))
    for i in (0, 2, 4, 6)
]


def upstream(obs, key):
    x = (obs["state"] - mean) / jnp.maximum(scale, 1e-8)
    for index, (weight, bias) in enumerate(layers):
        x = x @ weight.T + bias
        if index < len(layers) - 1:
            x = jax.nn.elu(x)
    return x


loaded = ppo_checkpoint.load_policy(sys.argv[1], network_factory=networks, deterministic=True)


def ours(obs, key):
    action, _ = loaded(obs, key)
    return action


def survey(stiff, policy, episodes=6, horizon=300):
    env = Velocity(spec=randomize.Spec(), noise=delay.Noise(), stiff=stiff)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    totals, steps = defaultdict(float), 0
    for seed in range(episodes):
        key = jax.random.key(seed)
        state = reset(key)
        for _ in range(horizon):
            key, act_key = jax.random.split(key)
            state = step(state, policy(state.obs, act_key))
            if float(state.done):
                break
            for name, value in state.metrics.items():
                totals[name.split("/")[-1]] += float(value)
            steps += 1
    return {name: value / max(steps, 1) for name, value in totals.items()}, steps


print(f"{'policy':>10s} {'physics':>8s} {'body_ang_vel':>13s} {'track_ang':>10s} {'upright':>8s} {'steps':>7s}")
for label, policy in (("ours", ours), ("upstream", upstream)):
    for stiff in (False, True):
        terms, steps = survey(stiff, policy)
        print(
            f"{label:>10s} {'stiff' if stiff else 'soft':>8s} "
            f"{terms.get('body_ang_vel', 0):13.4f} {terms.get('track_angular_velocity', 0):10.4f} "
            f"{terms.get('upright', 0):8.4f} {steps:7d}"
        )
