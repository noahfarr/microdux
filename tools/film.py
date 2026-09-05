import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

from microdux import Velocity, delay, randomize, render, sense
from microdux.train import networks

path = sys.argv[1]
out = sys.argv[2]
twist = jnp.asarray([float(v) for v in sys.argv[3].split(",")])
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 400

env = Velocity(spec=randomize.Spec(), noise=delay.Noise())
if path == "upstream":
    blob = np.load("/home/farr/microdux/tests/fixtures/upstream_policy.npz")
    mean = jnp.asarray(blob["obs_normalizer._mean"][0])
    scale = jnp.asarray(blob["obs_normalizer._std"][0])
    layers = [
        (jnp.asarray(blob[f"mlp.{i}.weight"]), jnp.asarray(blob[f"mlp.{i}.bias"]))
        for i in (0, 2, 4, 6)
    ]

    def loaded(obs, key):
        x = (obs["state"] - mean) / jnp.maximum(scale, 1e-8)
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
def hold(state):
    info = dict(state.info)
    info["commands"] = dataclasses.replace(state.info["commands"], twist=twist)
    return state.replace(info=info)


key = jax.random.key(0)
state = hold(reset(key))
frames, speeds, spins = [], [], []
for _ in range(steps):
    key, act_key = jax.random.split(key)
    action, _ = loaded(state.obs, act_key)
    state = hold(step(state, action))
    if float(state.done):
        break
    frames.append(np.asarray(state.data.qpos))
    speeds.append(float(sense.root_linear_velocity(state.data)[0]))
    spins.append(float(sense.root_angular_velocity(state.data)[2]))

qpos = np.stack(frames)
travel = np.linalg.norm(qpos[-1][:2] - qpos[0][:2])
print(f"command {np.asarray(twist)}  survived {len(qpos)}/{steps} steps")
print(f"forward {np.mean(speeds):+.3f} m/s   yaw {np.mean(spins):+.3f} rad/s")
print(f"trunk z: start {qpos[0][2]:.3f} min {qpos[:, 2].min():.3f} end {qpos[-1][2]:.3f}")
print(f"distance travelled {travel:.3f} m")
print("wrote", render.film(env, qpos, out))
