import json
import pathlib

import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from microdux import Velocity, commands, contact, delay, plant, rewards, sense

FIXTURE = pathlib.Path(__file__).parent.parent / "tests" / "fixtures" / "transition.json"


def state(env, frozen):
    seed = frozen["state"]
    qpos = np.asarray(env.mj_model.qpos0, dtype=np.float64).copy()
    qpos[:7] = seed["root"]
    qpos[env._wiring.qpos_adr] = seed["joint_pos"]
    qvel = np.zeros(env.mj_model.nv)
    qvel[:6] = seed["root_vel"]
    qvel[env._wiring.qvel_adr] = seed["joint_vel"]
    data = env._template.replace(qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel))
    return mjx.forward(env.mjx_model, data)


def info(env, frozen, data):
    seed = frozen["state"]
    zeros = jnp.zeros(env.mj_model.nu)
    rest = commands.rest(jnp.zeros(2, jnp.uint32), env._ranges, env._ctrl_dt)
    held = rest.replace(
        twist=jnp.asarray(seed["command"]),
        head=jnp.asarray(seed["head_pose"]),
        body=jnp.asarray(seed["body_pose"]),
    )
    return {
        "servos": plant.rest(env.mj_model.nu),
        "gait": contact.rest(len(env._feet)),
        "commands": held,
        "last_action": zeros,
        "previous_action": zeros,
        "head_bias": jnp.zeros(len(env._head_slots)),
        "targets": jnp.tile(env._home, (delay.ACTION.max_lag + 1, 1)),
        "draw": None,
    }


def ours(env, frozen):
    data = state(env, frozen)
    carried = info(env, frozen, data)
    action = jnp.asarray(frozen["state"]["action"])

    touching = contact.touching(data, env._feet)
    heights = data.site_xpos[env._foot_sites][:, 2]
    peak = carried["gait"].swing_peak
    gait, filtered, landed, air = contact.tally(
        carried["gait"], touching, heights, env._ctrl_dt
    )

    head = data.qpos[env._wiring.qpos_adr][env._head_slots]
    error = head - (env._home[env._head_slots] + carried["commands"].head)
    carried["head_bias"] = rewards.blend(
        carried["head_bias"], error, env._ctrl_dt, env._tuning.head_tau
    )

    terms = env._rewards(data, carried, action, touching, air, landed, peak, heights)
    weights = dict(vars(env._weights))
    return {name: float(terms[name]) * weights[name] for name in terms}, weights


def main():
    frozen = json.loads(FIXTURE.read_text())
    env = Velocity(envs=1, spec=None, noise=None)
    mine, _ = ours(env, frozen)
    theirs = frozen["terms"]

    print(f"{'term':26s} {'ours':>13s} {'upstream':>13s} {'delta':>12s}")
    worst, worst_name = 0.0, ""
    for name in sorted(theirs):
        a, b = mine.get(name), theirs[name]
        if a is None:
            print(f"{name:26s} {'absent':>13s} {b:13.6f}")
            continue
        delta = abs(a - b)
        if delta > worst:
            worst, worst_name = delta, name
        flag = "" if delta < 1e-4 else "   <-- differs"
        print(f"{name:26s} {a:13.6f} {b:13.6f} {delta:12.2e}{flag}")

    total = sum(mine.values()) * env._ctrl_dt
    print(f"\n{'total (dt scaled)':26s} {total:13.6f} {frozen['total']:13.6f} "
          f"{abs(total - frozen['total']):12.2e}")
    print(f"largest term disagreement {worst:.3e} on {worst_name or 'nothing'}")


if __name__ == "__main__":
    main()
