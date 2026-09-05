import argparse
import pathlib

import jax
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

from microdux import delay, randomize, render
from microdux.run import build
from microdux.train import networks


def latest(root: str) -> str:
    saved = sorted(p for p in pathlib.Path(root).iterdir() if p.is_dir())
    if not saved:
        raise SystemExit(f"no checkpoints under {root}")
    return str(saved[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--impl", default="jax")
    parser.add_argument("--shared-critic", action="store_true")
    args = parser.parse_args()

    env = build(args.task, 1, args.impl, nominal=False)
    path = pathlib.Path(args.checkpoints).expanduser().resolve()
    if not (path / "_METADATA").exists():
        path = pathlib.Path(latest(str(path)))
    path = str(path)

    policy = ppo_checkpoint.load_policy(
        path, network_factory=networks, deterministic=True
    )

    def act(obs, key):
        action, _ = policy(obs, key)
        return action

    qpos, rewards, dones = render.rollout(env, act, steps=args.steps, seed=args.seed)
    print(f"{args.task}: {len(qpos)} frames, reward {rewards.sum():.2f}, "
          f"terminated={bool(dones[-1])} from {path}", flush=True)
    render.film(env, qpos, args.out)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
