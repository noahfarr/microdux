import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from microdux import Velocity, delay, randomize, sense
from microdux.train import networks

params = pickle.load(open(sys.argv[1], "rb"))

env = Velocity(spec=randomize.Spec(), noise=delay.Noise())
sizes = {
    key: (value if isinstance(value, int) else value[-1])
    for key, value in env.observation_size.items()
}
network = networks(sizes, env.action_size, preprocess_observations_fn=running_statistics.normalize)
inference = ppo_networks.make_inference_fn(network)
policy = inference((params[0], params[1]), deterministic=True)

reset, step = jax.jit(env.reset), jax.jit(env.step)


@jax.jit
def measure(state):
    return sense.root_linear_velocity(state.data)[:2], sense.root_angular_velocity(state.data)[2]


rows = []
for seed in range(8):
    key = jax.random.key(seed)
    state = reset(key)
    for _ in range(400):
        key, act_key = jax.random.split(key)
        action, _ = policy(state.obs, act_key)
        state = step(state, action)
        if float(state.done):
            break
        lin, yaw = measure(state)
        command = np.asarray(state.info["commands"].twist)
        rows.append((np.asarray(lin), float(yaw), command[:2], float(command[2])))

lin = np.stack([r[0] for r in rows])
yaw = np.array([r[1] for r in rows])
cmd_lin = np.stack([r[2] for r in rows])
cmd_yaw = np.array([r[3] for r in rows])

print(f"samples {len(rows)}")
forward, cmd_forward = lin[:, 0], cmd_lin[:, 0]
moving = np.abs(cmd_forward) > 0.05
spinning = np.abs(cmd_yaw) > 0.2
print(f"forward fraction of command: "
      f"{(forward[moving] * np.sign(cmd_forward[moving])).mean() / np.abs(cmd_forward[moving]).mean():.2f}")
print(f"yaw fraction of command:     "
      f"{(yaw[spinning] * np.sign(cmd_yaw[spinning])).mean() / np.abs(cmd_yaw[spinning]).mean():.2f}")
