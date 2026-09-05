import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np

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
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    loaded = ppo_checkpoint.load_policy(path, network_factory=networks, deterministic=True)

reset = jax.jit(env.reset)
step = jax.jit(env.step)


@jax.jit
def hold(state, twist):
    held = dataclasses.replace(state.info["commands"], twist=twist)
    info = dict(state.info)
    info["commands"] = held
    return state.replace(info=info)


@jax.jit
def measure(state):
    return sense.root_linear_velocity(state.data)[:2], sense.root_angular_velocity(state.data)[2]


def sweep(twist, seeds=6, horizon=250, settle=60):
    speeds, spins, jerk = [], [], []
    twist = jnp.asarray(twist)
    for seed in range(seeds):
        key = jax.random.key(seed)
        state = hold(reset(key), twist)
        previous = None
        for tick in range(horizon):
            key, act_key = jax.random.split(key)
            action, _ = loaded(state.obs, act_key)
            state = hold(step(state, action), twist)
            if float(state.done):
                break
            if tick >= settle:
                lin, yaw = measure(state)
                speeds.append(float(lin[0]))
                spins.append(float(yaw))
                if previous is not None:
                    jerk.append(float(jnp.mean(jnp.square(action - previous))))
            previous = action
    return (
        float(np.mean(speeds)) if speeds else float("nan"),
        float(np.mean(spins)) if spins else float("nan"),
        float(np.mean(jerk)) if jerk else float("nan"),
        len(speeds),
    )


print(f"{'command':>22s} {'fwd m/s':>9s} {'yaw rad/s':>10s} {'jerk':>8s} {'n':>6s}")
for label, twist in (
    ("stand", (0.0, 0.0, 0.0)),
    ("fwd 0.10", (0.10, 0.0, 0.0)),
    ("fwd 0.20", (0.20, 0.0, 0.0)),
    ("fwd 0.30", (0.30, 0.0, 0.0)),
    ("back -0.15", (-0.15, 0.0, 0.0)),
    ("yaw +0.5", (0.0, 0.0, 0.5)),
    ("yaw +1.0", (0.0, 0.0, 1.0)),
    ("yaw -1.0", (0.0, 0.0, -1.0)),
):
    fwd, yaw, jerk, n = sweep(twist)
    print(f"{label:>22s} {fwd:9.3f} {yaw:10.3f} {jerk:8.4f} {n:6d}")
