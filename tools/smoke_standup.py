import sys
sys.path.insert(0, "/home/farr/microdux")
import jax
import jax.numpy as jnp
from microdux import StandUp, SitStand

for name, cls in (("StandUp", StandUp), ("SitStand", SitStand)):
    print("===", name, "===")
    env = cls()
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.key(0))
    print("obs state shape", state.obs["state"].shape)
    print("obs privileged shape", state.obs["privileged_state"].shape)
    for i in range(5):
        state = step(state, jnp.zeros(env.action_size))
        print(i, "reward", float(state.reward), "done", float(state.done))
    print("qpos finite", bool(jnp.isfinite(state.data.qpos).all()))
