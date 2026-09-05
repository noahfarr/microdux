import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics, specs

from microdux import Velocity
from microdux.train import networks

out = sys.argv[1]
source = sys.argv[2] if len(sys.argv) > 2 else "/home/farr/microdux/tests/fixtures/upstream_policy.npz"

env = Velocity()
sizes = {
    key: (value if isinstance(value, int) else value[-1])
    for key, value in env.observation_size.items()
}
network = networks(sizes, env.action_size, preprocess_observations_fn=running_statistics.normalize)

key = jax.random.key(0)
policy_key, value_key = jax.random.split(key)
policy = network.policy_network.init(policy_key)
value = network.value_network.init(value_key)

blob = np.load(source)
weights = policy["params"]
for ours, theirs in (("hidden_0", 0), ("hidden_1", 2), ("hidden_2", 4)):
    weights["MLP_0"][ours]["kernel"] = jnp.asarray(blob[f"mlp.{theirs}.weight"]).T
    weights["MLP_0"][ours]["bias"] = jnp.asarray(blob[f"mlp.{theirs}.bias"])
weights["Dense_0"]["kernel"] = jnp.asarray(blob["mlp.6.weight"]).T
weights["Dense_0"]["bias"] = jnp.asarray(blob["mlp.6.bias"])
weights["std_param"]["value"] = jnp.asarray(blob["distribution.std_param"])

mean = jnp.asarray(blob["obs_normalizer._mean"][0])
std = jnp.asarray(blob["obs_normalizer._std"][0])
var = jnp.asarray(blob["obs_normalizer._var"][0])
count = float(blob["obs_normalizer.count"])


def widen(vector, width, fill):
    grown = jnp.full((width,), fill, vector.dtype)
    return grown.at[: vector.shape[0]].set(vector)


normaliser = running_statistics.init_state(
    jax.tree.map(lambda size: specs.Array((size,), jnp.dtype("float32")), sizes)
)
normaliser = normaliser.replace(
    mean={key: widen(mean, sizes[key], 0.0) for key in sizes},
    std={key: widen(std, sizes[key], 1.0) for key in sizes},
    summed_variance={key: widen(var * count, sizes[key], count) for key in sizes},
)

params = (normaliser, policy, value)
with open(out, "wb") as handle:
    pickle.dump(jax.device_get(params), handle)
print(f"wrote {out}")
print(f"  std_param  {np.asarray(blob['distribution.std_param'])[:4]} ...")
print(f"  obs count  {count:.0f}")
print(f"  sizes      {sizes}")
