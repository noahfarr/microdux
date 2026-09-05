import functools
import time

import jax
import jax.numpy as jnp
import numpy as np


def _replicate(value, devices):
    stacked = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (len(devices),) + jnp.shape(leaf)), value
    )
    if len(devices) == 1:
        return jax.device_put(stacked, devices[0])

    mesh = jax.sharding.Mesh(np.asarray(devices), ("device",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("device"))
    return jax.device_put(stacked, sharding)


try:
    jax.device_put_replicated
except AttributeError:
    jax.device_put_replicated = _replicate

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import optimizer as ppo_optimizer
from brax.training.agents.ppo import train as ppo
from brax.envs.wrappers import training as brax_wrapper
from mujoco_playground import wrapper

from . import wrappers

from .delay import Noise
from .env import Velocity
from .randomize import Spec

HIDDEN = (512, 256, 128)


def keep(params, path):
    import pathlib
    import pickle

    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pickle.dumps(jax.device_get(params)))
    return target


def revive(path):
    import pickle

    return pickle.loads(open(path, "rb").read())


def policy_from(env, params, deterministic: bool = True):
    from brax.training.acme import running_statistics, specs

    sizes = jax.tree.map(lambda leaf: leaf.shape[-1], env.observation_size)
    normalise = running_statistics.normalize
    network = networks(sizes, env.action_size, preprocess_observations_fn=normalise)
    inference = ppo_networks.make_inference_fn(network)
    apply = inference(params, deterministic=deterministic)

    def act(obs, key):
        action, _ = apply(obs, key)
        return action

    return act


def harness(env, episode_length: int = 1000, action_repeat: int = 1,
            randomization_fn=None, full_reset: bool = False):
    if randomization_fn is None:
        env = brax_wrapper.VmapWrapper(env)
    else:
        env = wrapper.BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    env = brax_wrapper.EpisodeWrapper(env, episode_length, action_repeat)
    env = wrappers.AutoReset(env) if full_reset else wrapper.BraxAutoResetWrapper(env)
    return wrappers.Bootstrapping(env, episode_length)


def networks(*args, privileged: bool = True, bounded: bool = False, **kwargs):
    return ppo_networks.make_ppo_networks(
        *args,
        policy_hidden_layer_sizes=HIDDEN,
        value_hidden_layer_sizes=HIDDEN,
        activation=jax.nn.elu,
        distribution_type="tanh_normal" if bounded else "normal",
        noise_std_type="scalar",
        init_noise_std=1.0,
        value_obs_key="privileged_state" if privileged else "state",
        **kwargs,
    )


def learn(
    env=None,
    eval_env=None,
    timesteps: int = 30_000_000,
    num_envs: int = 2048,
    unroll_length: int = 24,
    batch_size: int = 256,
    num_minibatches: int = 4,
    updates_per_batch: int = 5,
    learning_rate: float = 1e-3,
    entropy_cost: float = 0.01,
    discounting: float = 0.99,
    lambda_: float = 0.95,
    clipping_epsilon: float = 0.2,
    episode_length: int = 1000,
    eval_envs: int = 32,
    seed: int = 0,
    progress=None,
    checkpoints: str | None = None,
    privileged: bool = True,
    bounded: bool = False,
    adaptive: bool = True,
    desired_kl: float = 0.01,
    max_grad_norm: float = 1.0,
    nominal: bool = False,
    restore_params=None,
):
    if env is None:
        env = Velocity(
            envs=num_envs,
            spec=None if nominal else Spec(),
            noise=None if nominal else Noise(),
        )
    started = time.time()
    history = []

    def report(step, metrics):
        prefix = "eval/" if "eval/avg_episode_length" in metrics else "training/"
        if prefix == "training/" and "training/episode_length" not in metrics:
            return
        length = float(
            metrics.get(f"{prefix}avg_episode_length")
            or metrics.get("training/episode_length", float("nan"))
        )
        terms = {
            key.rsplit("/", 1)[-1]: float(value) / max(length, 1.0)
            for key, value in metrics.items()
            if key.startswith(f"{prefix}episode_reward/")
        }
        entry = {
            "step": int(step),
            "reward": float(
                metrics.get(f"{prefix}episode_reward")
                or metrics.get("training/episode_reward", float("nan"))
            ),
            "length": length,
            "minutes": (time.time() - started) / 60.0,
            "terms": terms,
        }
        history.append(entry)
        walking = terms.get("air_time", float("nan"))
        tracking = terms.get("track_linear_velocity", float("nan"))
        rate = float(metrics.get("training/learning_rate", float("nan")))
        divergence = float(metrics.get("training/kl_mean", float("nan")))
        entry["learning_rate"] = rate
        entry["kl"] = divergence
        print(
            f"{entry['step']:>10,} steps  reward {entry['reward']:8.3f}  "
            f"episode {entry['length']:7.1f}  air {walking:6.3f}  "
            f"track {tracking:5.3f}  lr {rate:.1e}  kl {divergence:.4f}  "
            f"{entry['minutes']:5.1f} min",
            flush=True,
        )
        if progress is not None:
            progress(entry, metrics)

    make_policy, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        num_timesteps=timesteps,
        num_evals=max(2, timesteps // 10_000_000),
        run_evals=False,
        episode_length=episode_length,
        num_envs=num_envs,
        num_eval_envs=eval_envs,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_updates_per_batch=updates_per_batch,
        unroll_length=unroll_length,
        learning_rate=learning_rate,
        learning_rate_schedule=(
            ppo_optimizer.LRSchedule.ADAPTIVE_KL if adaptive else None
        ),
        desired_kl=desired_kl,
        entropy_cost=entropy_cost,
        discounting=discounting,
        gae_lambda=lambda_,
        clipping_epsilon=clipping_epsilon,
        max_grad_norm=max_grad_norm,
        normalize_observations=True,
        network_factory=functools.partial(networks, privileged=privileged, bounded=bounded),
        wrap_env_fn=functools.partial(harness, full_reset=True),
        seed=seed,
        progress_fn=report,
        log_training_metrics=True,
        training_metrics_steps=2_000_000,
        restore_params=restore_params,
        save_checkpoint_path=checkpoints,
    )
    if checkpoints:
        keep(params, f"{checkpoints}/final.pkl")
    return make_policy, params, history


def widen(vector, width, fill):
    shared = min(vector.shape[0], width)
    grown = jnp.full((width,), fill, vector.dtype)
    return grown.at[:shared].set(vector[:shared])


def stretch(kernel, width, shared=None):
    rows = min(kernel.shape[0], width if shared is None else shared)
    grown = jnp.zeros((width, kernel.shape[1]), kernel.dtype)
    return grown.at[:rows].set(kernel[:rows])


def graft(source, sizes):
    normaliser, policy, value = source[0], source[1], source[2]
    shared = int(normaliser.mean["state"].shape[0])
    if shared == sizes["state"]:
        return source

    tally = normaliser.count
    count = jnp.maximum(
        jnp.asarray(tally.hi, jnp.float32) * 2.0**32 + jnp.asarray(tally.lo, jnp.float32),
        1.0,
    )
    normaliser = normaliser.replace(
        mean={key: widen(normaliser.mean["state"], sizes[key], 0.0) for key in sizes},
        std={key: widen(normaliser.std["state"], sizes[key], 1.0) for key in sizes},
        summed_variance={
            key: widen(normaliser.summed_variance["state"], sizes[key], count)
            for key in sizes
        },
    )

    policy = jax.tree_util.tree_map(lambda leaf: leaf, policy)
    policy["params"]["MLP_0"]["hidden_0"]["kernel"] = stretch(
        policy["params"]["MLP_0"]["hidden_0"]["kernel"], sizes["state"]
    )
    value = jax.tree_util.tree_map(lambda leaf: leaf, value)
    value["params"]["hidden_0"]["kernel"] = stretch(
        value["params"]["hidden_0"]["kernel"], sizes["privileged_state"], shared
    )
    return (normaliser, policy, value)
