import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

SHARED = 61


def fold(kernel, bias, mean, std, width):
    """Bake an affine observation normalizer into the first layer.

    brax normalizes inside the network, relax in a wrapper, so a policy that
    moves between them has to carry its normalizer in its weights instead.
    """
    scaled = kernel / std[:, None]
    shifted = bias - (mean / std) @ kernel
    grown = jnp.zeros((width, kernel.shape[1]), kernel.dtype)
    return grown.at[: kernel.shape[0]].set(scaled), shifted


def convert(source, obs_width: int, actions: int):
    normaliser, policy, value = source[0], source[1], source[2]
    mean = normaliser.mean["state"]
    std = jnp.maximum(normaliser.std["state"], 1e-8)

    trunk = policy["params"]["MLP_0"]
    head = policy["params"]["Dense_0"]
    spread = policy["params"]["std_param"]["value"]

    kernel, bias = fold(
        trunk["hidden_0"]["kernel"], trunk["hidden_0"]["bias"], mean, std, obs_width
    )
    actor = {
        "layers_0": {"kernel": kernel, "bias": bias},
        "layers_2": {
            "kernel": trunk["hidden_1"]["kernel"],
            "bias": trunk["hidden_1"]["bias"],
        },
        "layers_4": {
            "kernel": trunk["hidden_2"]["kernel"],
            "bias": trunk["hidden_2"]["bias"],
        },
    }

    # relax splits the head into mean and a softplus scale; brax keeps the
    # scale as a free parameter, so it becomes a bias with zero weights.
    raw = jnp.log(jnp.expm1(jnp.maximum(spread - 1e-3, 1e-4)))
    actor["layers_6"] = {
        "kernel": jnp.concatenate(
            [head["kernel"], jnp.zeros((head["kernel"].shape[0], actions))], axis=-1
        ),
        "bias": jnp.concatenate([head["bias"], raw]),
    }

    critic_mean = normaliser.mean["privileged_state"][:SHARED]
    critic_std = jnp.maximum(normaliser.std["privileged_state"][:SHARED], 1e-8)
    stem = value["params"]
    kernel, bias = fold(
        stem["hidden_0"]["kernel"][:SHARED],
        stem["hidden_0"]["bias"],
        critic_mean,
        critic_std,
        obs_width,
    )
    critic = {
        "layers_0": {"kernel": kernel, "bias": bias},
        "layers_2": {"kernel": stem["hidden_1"]["kernel"], "bias": stem["hidden_1"]["bias"]},
        "layers_4": {"kernel": stem["hidden_2"]["kernel"], "bias": stem["hidden_2"]["bias"]},
        "layers_6": {"kernel": stem["hidden_3"]["kernel"], "bias": stem["hidden_3"]["bias"]},
    }

    return {"params": {"module": {"head": {"actor": actor, "critic": critic}}}}


if __name__ == "__main__":
    checkpoint, out = sys.argv[1], sys.argv[2]
    obs_width = int(sys.argv[3]) if len(sys.argv) > 3 else 73
    actions = int(sys.argv[4]) if len(sys.argv) > 4 else 14

    tree = convert(ppo_checkpoint.load(checkpoint), obs_width, actions)
    with open(out, "wb") as handle:
        pickle.dump(jax.device_get(tree), handle)

    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        print(f"{jax.tree_util.keystr(path):58s} {np.shape(leaf)}")
    print("wrote", out)
