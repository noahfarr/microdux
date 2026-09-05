import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from microdux import Velocity, delay, randomize
from microdux.train import networks

source, out = sys.argv[1], sys.argv[2]
envs = int(sys.argv[3]) if len(sys.argv) > 3 else 256
horizon = int(sys.argv[4]) if len(sys.argv) > 4 else 400
epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 400
gamma = 0.99

params = pickle.load(open(source, "rb"))
env = Velocity(envs=envs, spec=randomize.Spec(), noise=delay.Noise())
sizes = {
    key: (value if isinstance(value, int) else value[-1])
    for key, value in env.observation_size.items()
}
network = networks(sizes, env.action_size, preprocess_observations_fn=running_statistics.normalize)
inference = ppo_networks.make_inference_fn(network)
policy = inference((params[0], params[1]), deterministic=False)

reset = jax.jit(jax.vmap(env.reset))
step = jax.jit(jax.vmap(env.step))

keys = jax.random.split(jax.random.key(0), envs)
state = reset(keys)

observations, rewards, dones = [], [], []
key = jax.random.key(1)
for _ in range(horizon):
    key, act_key = jax.random.split(key)
    action, _ = policy(state.obs, act_key)
    observations.append(state.obs["privileged_state"])
    state = step(state, action)
    rewards.append(state.reward)
    dones.append(state.done)

observations = jnp.stack(observations)
rewards = jnp.stack(rewards)
dones = jnp.stack(dones)
print(f"collected {observations.shape} obs, mean reward {float(rewards.mean()):.4f}")


def discount(carry, row):
    reward, done = row
    carry = reward + gamma * carry * (1.0 - done)
    return carry, carry


_, returns = jax.lax.scan(
    discount, jnp.zeros(envs), (rewards[::-1], dones[::-1])
)
returns = returns[::-1]
print(f"returns mean {float(returns.mean()):.3f} std {float(returns.std()):.3f}")

flat_obs = observations.reshape(-1, observations.shape[-1])
flat_returns = returns.reshape(-1)

value_params = params[2]
optimiser = optax.adam(3e-4)
opt_state = optimiser.init(value_params)
normaliser = params[0]


def loss(value_params, obs, target):
    predicted = network.value_network.apply(normaliser, value_params, {"privileged_state": obs})
    return jnp.mean(jnp.square(predicted - target))


@jax.jit
def update(value_params, opt_state, obs, target):
    cost, grads = jax.value_and_grad(loss)(value_params, obs, target)
    updates, opt_state = optimiser.update(grads, opt_state)
    return optax.apply_updates(value_params, updates), opt_state, cost


batch = 4096
size = flat_obs.shape[0]
for epoch in range(epochs):
    key, pick = jax.random.split(key)
    index = jax.random.randint(pick, (batch,), 0, size)
    value_params, opt_state, cost = update(
        value_params, opt_state, flat_obs[index], flat_returns[index]
    )
    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"  epoch {epoch:4d} value loss {float(cost):.4f}")

with open(out, "wb") as handle:
    pickle.dump(jax.device_get((params[0], params[1], value_params)), handle)
print("wrote", out)
