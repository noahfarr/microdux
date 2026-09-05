import sys

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

from microdux import Velocity, delay, randomize, sense
from microdux.train import networks

env = Velocity(spec=randomize.Spec(), noise=delay.Noise())
reset = jax.jit(env.reset)
step = jax.jit(env.step)

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


for label, policy in (("ours", ours), ("upstream", upstream)):
    flights, peaks, contacts, speeds = [], [], [], []
    for seed in range(4):
        key = jax.random.key(seed)
        state = reset(key)
        previous = np.array([True, True])
        airborne = np.zeros(2)
        for _ in range(400):
            key, act_key = jax.random.split(key)
            state = step(state, policy(state.obs, act_key))
            if float(state.done):
                break
            touching = np.asarray(state.info["gait"].last_contact)
            air = np.asarray(state.info["gait"].air_time)
            landed = touching & ~previous
            for foot in range(2):
                if landed[foot] and airborne[foot] > 0:
                    flights.append(airborne[foot])
            airborne = np.where(touching, 0.0, air)
            previous = touching
            contacts.append(touching.sum())
            peaks.append(float(np.asarray(state.info["gait"].swing_peak).max()))
            speeds.append(float(jnp.linalg.norm(sense.root_linear_velocity(state.data)[:2])))

    flights = np.array(flights) if flights else np.array([0.0])
    print(f"\n{label}")
    print(f"  completed flights   : {len(flights)}")
    print(f"  flight duration (s) : mean {flights.mean():.3f}  median {np.median(flights):.3f}  max {flights.max():.3f}")
    print(f"  in reward window    : {((flights > 0.125) & (flights < 0.300)).mean() * 100:.0f}%")
    print(f"  feet in contact     : mean {np.mean(contacts):.2f} of 2")
    print(f"  swing peak (m)      : mean {np.mean(peaks):.4f}")
    print(f"  body speed (m/s)    : mean {np.mean(speeds):.3f}")
