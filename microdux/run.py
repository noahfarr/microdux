import argparse
import json
import pathlib
import time

from .delay import Noise
from .randomize import Spec
from .train import keep, learn, revive

TASKS = (
    "velocity", "rough", "velstand", "standup", "sitstand", "spin", "swizzle",
    "rollers", "rollercrouch", "rollerstandup", "rollerslope", "roulade",
    "ballkick", "groundpick",
)


def build(task: str, envs: int, impl: str, nominal: bool):
    import microdux
    from microdux import terrain

    shared = dict(
        envs=envs,
        impl=impl,
        spec=None if nominal else Spec(),
        noise=None if nominal else Noise(),
    )
    if task == "velocity":
        return microdux.Velocity(**shared)
    if task == "rough":
        return microdux.Velocity(terrain=terrain.Config(), **shared)
    named = {
        "velstand": "VelStand", "standup": "StandUp", "sitstand": "SitStand",
        "spin": "Spin", "swizzle": "Swizzle", "rollers": "Rollers",
        "rollercrouch": "RollerCrouch", "rollerstandup": "RollerStandUp",
        "rollerslope": "RollerSlope", "roulade": "Roulade",
        "ballkick": "BallKick", "groundpick": "GroundPick",
    }
    return getattr(microdux, named[task])(**shared)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="velocity", choices=TASKS)
    parser.add_argument("--impl", default="jax", choices=("jax", "warp"))
    parser.add_argument("--timesteps", type=int, default=200_000_000)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/velocity")
    parser.add_argument("--bounded", action="store_true")
    parser.add_argument("--shared-critic", action="store_true")
    parser.add_argument("--fixed-lr", action="store_true")
    parser.add_argument("--nominal", action="store_true")
    parser.add_argument("--warmstart", default=None)
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = out / "history.jsonl"
    started = time.time()

    def note(entry, metrics):
        with history.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")

    make_policy, params, log = learn(
        env=build(args.task, args.num_envs, args.impl, args.nominal),
        eval_env=build(args.task, args.eval_envs, args.impl, args.nominal),
        timesteps=args.timesteps,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        eval_envs=args.eval_envs,
        seed=args.seed,
        progress=note,
        checkpoints=str(out / "checkpoints"),
        privileged=not args.shared_critic,
        bounded=args.bounded,
        adaptive=not args.fixed_lr,
        nominal=args.nominal,
        restore_params=revive(args.warmstart) if args.warmstart else None,
    )

    keep(params, out / "policy.pkl")
    print(f"finished in {(time.time() - started) / 60:.1f} min -> {out / 'policy.pkl'}")


if __name__ == "__main__":
    main()
