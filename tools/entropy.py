import sys

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

from microdux import Velocity
from microdux.train import networks

env = Velocity()
sizes = {k: (v if isinstance(v, int) else v[-1]) for k, v in env.observation_size.items()}
network = networks(sizes, env.action_size, preprocess_observations_fn=running_statistics.normalize)

for path in sys.argv[1:]:
    params = ppo_checkpoint.load(path)
    normaliser, policy = params[0], params[1]
    obs = {"state": jnp.zeros(sizes["state"]), "privileged_state": jnp.zeros(sizes["privileged_state"])}
    raw = network.policy_network.apply(normaliser, policy, obs)
    dist = network.parametric_action_distribution
    loc, scale = raw if isinstance(raw, tuple) else jnp.split(raw, 2, axis=-1)
    scale = jnp.asarray(scale)
    entropy = float(jnp.sum(0.5 * jnp.log(2 * jnp.pi * jnp.e * scale**2)))
    print(f"== {path}")
    print(f"   scale param  min {float(scale.min()):.4f} mean {float(scale.mean()):.4f} max {float(scale.max()):.4f}")
    print(f"   entropy {entropy:.3f}  (14 dims)")
    implied = float(np.exp(entropy / 14 - 0.5 * np.log(2 * np.pi * np.e)))
    print(f"   implied std {implied:.4f}")
