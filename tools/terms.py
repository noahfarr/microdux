import sys
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

from microdux import Velocity, delay, randomize, rewards
from microdux.train import networks

FIXTURE = "/home/farr/microdux/tests/fixtures/upstream_policy.npz"

env = Velocity(spec=randomize.Spec(), noise=delay.Noise())
reset = jax.jit(env.reset)
step = jax.jit(env.step)

blob = np.load(FIXTURE)
mean = jnp.asarray(blob["obs_normalizer._mean"][0])
std = jnp.asarray(blob["obs_normalizer._std"][0])
layers = [
    (jnp.asarray(blob[f"mlp.{i}.weight"]), jnp.asarray(blob[f"mlp.{i}.bias"]))
    for i in (0, 2, 4, 6)
]


def upstream(obs, key):
    x = (obs["state"] - mean) / jnp.maximum(std, 1e-8)
    for index, (weight, bias) in enumerate(layers):
        x = x @ weight.T + bias
        if index < len(layers) - 1:
            x = jax.nn.elu(x)
    return x


loaded = ppo_checkpoint.load_policy(sys.argv[1], network_factory=networks, deterministic=True)


def ours(obs, key):
    action, _ = loaded(obs, key)
    return action


def tally(policy, episodes=4, horizon=400):
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
    return {name: value / steps for name, value in totals.items()}, steps


mine, mine_steps = tally(ours)
theirs, their_steps = tally(upstream)
weights = vars(rewards.Weights())

print(f"{'term':26s} {'ours':>9s} {'upstream':>9s} {'weight':>8s} {'delta*w':>9s}")
rows = []
for name in sorted(set(mine) | set(theirs)):
    a, b = mine.get(name, 0.0), theirs.get(name, 0.0)
    w = weights.get(name, 0.0)
    rows.append((name, a, b, w, (a - b) * w))
for name, a, b, w, delta in sorted(rows, key=lambda r: -abs(r[4])):
    print(f"{name:26s} {a:9.4f} {b:9.4f} {w:8.3f} {delta:9.4f}")

print(f"\nour net per step      {sum(r[1] * r[3] for r in rows):.4f}")
print(f"upstream net per step {sum(r[2] * r[3] for r in rows):.4f}")
print(f"steps: ours {mine_steps}, upstream {their_steps}")
