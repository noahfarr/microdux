import jax
import jax.numpy as jnp
from mujoco_playground._src import mjx_env
from mujoco_playground._src import wrapper

PRESERVE = "AutoResetWrapper_preserve_info"


def innermost(env):
    while hasattr(env, "env"):
        env = env.env
    return env


class Bootstrapping(wrapper.Wrapper):
    def __init__(self, env, episode_length: int):
        super().__init__(env)
        self._episode_length = episode_length

    def step(self, state, action):
        stepped = self.env.step(state, action)
        info = dict(stepped.info)
        info["truncation"] = jnp.where(
            (info["steps"] >= self._episode_length) & (stepped.done > 0),
            jnp.ones_like(stepped.done),
            jnp.zeros_like(stepped.done),
        )
        return stepped.replace(info=info)


class AutoReset(wrapper.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._key = "AutoResetWrapper"
        self._respawn = getattr(innermost(env), "respawn", None)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        split = jax.vmap(jax.random.split)(rng)
        rng, key = split[..., 0], split[..., 1]
        state = self.env.reset(key)
        state.info[f"{self._key}_rng"] = rng
        state.info[f"{self._key}_done_count"] = jnp.zeros(key.shape[:-1], dtype=int)
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        split = jax.vmap(jax.random.split)(state.info[f"{self._key}_rng"])
        reset_rng, reset_key = split[..., 0], split[..., 1]

        reset_state = self.reset(reset_key)
        if self._respawn is not None:
            fresh = jax.vmap(self._respawn)(
                reset_state, state.info[PRESERVE], reset_key
            )
            info = dict(reset_state.info)
            info.update(fresh.info)
            reset_state = reset_state.replace(
                data=fresh.data, obs=fresh.obs, info=info
            )

        if "steps" in state.info:
            steps = state.info["steps"]
            state.info.update(steps=jnp.where(state.done, jnp.zeros_like(steps), steps))

        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)

        data = jax.tree.map(where_done, reset_state.data, state.data)
        obs = jax.tree.map(where_done, reset_state.obs, state.obs)

        info = jax.tree.map(where_done, reset_state.info, state.info)
        info[f"{self._key}_done_count"] = state.info[f"{self._key}_done_count"]
        if "steps" in info:
            info["steps"] = state.info["steps"]
        info[PRESERVE] = state.info[PRESERVE]
        info[f"{self._key}_done_count"] += state.done.astype(int)
        info[f"{self._key}_rng"] = reset_rng

        return state.replace(data=data, obs=obs, info=info)
