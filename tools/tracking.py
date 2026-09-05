import sys

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

from microdux import Velocity, delay, randomize, sense
from microdux.train import networks

path = sys.argv[1]
env = Velocity(spec=randomize.Spec(), noise=delay.Noise())

if path == "upstream":
    blob = np.load("/home/farr/microdux/tests/fixtures/upstream_policy.npz")
    mean = jnp.asarray(blob["obs_normalizer._mean"][0])
    std = jnp.asarray(blob["obs_normalizer._std"][0])
    layers = [
        (jnp.asarray(blob[f"mlp.{i}.weight"]), jnp.asarray(blob[f"mlp.{i}.bias"]))
        for i in (0, 2, 4, 6)
    ]

    def loaded(obs, key):
        x = (obs["state"] - mean) / jnp.maximum(std, 1e-8)
        for index, (weight, bias) in enumerate(layers):
            x = x @ weight.T + bias
            if index < len(layers) - 1:
                x = jax.nn.elu(x)
        return x, None
else:
    loaded = ppo_checkpoint.load_policy(path, network_factory=networks, deterministic=True)

reset = jax.jit(env.reset)
step = jax.jit(env.step)


@jax.jit
def measure(state):
    body = sense.root_linear_velocity(state.data)
    spin = sense.root_angular_velocity(state.data)
    return body[:2], spin[2]


rows = []
for seed in range(8):
    key = jax.random.key(seed)
    state = reset(key)
    for _ in range(400):
        key, act_key = jax.random.split(key)
        action, _ = loaded(state.obs, act_key)
        state = step(state, action)
        if float(state.done):
            break
        actual_lin, actual_yaw = measure(state)
        command = np.asarray(state.info["commands"].twist)
        rows.append(
            (np.asarray(actual_lin), float(actual_yaw), command[:2], float(command[2]))
        )

lin = np.stack([r[0] for r in rows])
yaw = np.array([r[1] for r in rows])
cmd_lin = np.stack([r[2] for r in rows])
cmd_yaw = np.array([r[3] for r in rows])

print(f"samples: {len(rows)}")
print(f"commanded |lin| mean {np.linalg.norm(cmd_lin, axis=1).mean():.3f}  "
      f"achieved |lin| mean {np.linalg.norm(lin, axis=1).mean():.3f}")
print(f"commanded yaw  mean {np.abs(cmd_yaw).mean():.3f}  "
      f"achieved yaw  mean {np.abs(yaw).mean():.3f}")

lin_err = np.linalg.norm(lin - cmd_lin, axis=1)
yaw_err = np.abs(yaw - cmd_yaw)
print(f"linear tracking error  mean {lin_err.mean():.3f} m/s")
print(f"angular tracking error mean {yaw_err.mean():.3f} rad/s")

forward = lin[:, 0]
cmd_forward = cmd_lin[:, 0]
moving = np.abs(cmd_forward) > 0.05
if moving.sum():
    ratio = (forward[moving] * np.sign(cmd_forward[moving])).mean() / np.abs(cmd_forward[moving]).mean()
    print(f"forward speed as fraction of forward command: {ratio:.2f} (n={moving.sum()})")

spinning = np.abs(cmd_yaw) > 0.2
if spinning.sum():
    ratio = (yaw[spinning] * np.sign(cmd_yaw[spinning])).mean() / np.abs(cmd_yaw[spinning]).mean()
    print(f"yaw rate as fraction of yaw command:          {ratio:.2f} (n={spinning.sum()})")
