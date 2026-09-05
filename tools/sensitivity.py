import sys

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

from microdux import Velocity, delay, randomize
from microdux.train import networks

TWIST = slice(48, 51)

env = Velocity(spec=randomize.Spec(), noise=delay.Noise())
state = jax.jit(env.reset)(jax.random.key(0))

blob = np.load("/home/farr/microdux/tests/fixtures/upstream_policy.npz")
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


ours_loaded = ppo_checkpoint.load_policy(
    sys.argv[1], network_factory=networks, deterministic=True
)


def ours(obs, key):
    action, _ = ours_loaded(obs, key)
    return action


base = state.obs["state"]
key = jax.random.key(0)

for label, policy in (("ours", ours), ("upstream", upstream)):
    forward = base.at[TWIST].set(jnp.array([0.4, 0.0, 0.0]))
    backward = base.at[TWIST].set(jnp.array([-0.4, 0.0, 0.0]))
    left = base.at[TWIST].set(jnp.array([0.0, 0.0, 1.0]))
    right = base.at[TWIST].set(jnp.array([0.0, 0.0, -1.0]))
    still = base.at[TWIST].set(jnp.zeros(3))

    def run(obs):
        return np.asarray(policy({"state": obs, "privileged_state": state.obs["privileged_state"]}, key))

    a_f, a_b = run(forward), run(backward)
    a_l, a_r = run(left), run(right)
    a_s = run(still)

    print(f"\n{label}")
    print(f"  |action(fwd) - action(back)| : {np.abs(a_f - a_b).mean():.4f}")
    print(f"  |action(left) - action(right)|: {np.abs(a_l - a_r).mean():.4f}")
    print(f"  |action(fwd) - action(still)| : {np.abs(a_f - a_s).mean():.4f}")
    print(f"  action magnitude              : {np.abs(a_s).mean():.4f}")
