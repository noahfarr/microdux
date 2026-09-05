# Porting an upstream task into microdux

## What microdux is

A JAX/MJX port of `pollen-robotics/microduck_rl`, which trains the Microduck on
mjlab (torch on MuJoCo Warp). microdux holds upstream's own tasks plus the
machinery to build new ones. Games built on top of it live elsewhere.

## Where things are

- This repo: `/home/farr/microdux`, but **work only inside your own worktree**.
- Upstream source to port from:
  `/tmp/claude-1356/-home-farr-relax/6056069a-ed57-4295-b42d-c6304b8d1da7/scratchpad/microduck_rl/src/mjlab_microduck/`
  Task configs are `tasks/microduck_<name>_env_cfg.py`, shared terms are
  `tasks/mdp.py`.
- Python: `/home/farr/microdux/.venv/bin/python`, with `PYTHONPATH` set to your
  worktree so your edits are what runs. Check with `print(microdux.__file__)`.

## The pattern to follow

Read `microdux/env.py` (`Velocity`) for a ground task and `microdux/spin.py` for
a roller task. A task is one class deriving `mjx_env.MjxEnv` with `reset`,
`step`, `_observe`, `_rewards`, `_measure`, and `struct.dataclass` `Weights` and
`Tuning` holding every constant. Reward terms are pure functions in
`microdux/rewards.py`; add new ones there rather than inlining them.

Curricula are step-keyed staircases in `microdux/curricula.py`. Upstream's
`{"step": N, "weight": W}` schedules become tuples of `(step, value)` pairs read
with `curricula.staircase`.

## Rules that are not negotiable

1. **No comments and no docstrings.** The codebase has none. Names carry the
   meaning. Use short evocative verbs for functions.
2. **`reset` must not call `mjx_env.make_data`.** Build one template in
   `__init__`:
   ```python
   self._template = mjx_env.make_data(
       self._mj_model, impl=impl,
       naconmax=self._nconmax * envs, njmax=self._njmax,
   )
   ```
   and in `reset` do `self._template.replace(qpos=qpos, qvel=qvel)`. Warp's
   contact arrays are allocated across all worlds, so calling `make_data` inside
   a vmapped reset gives every world a single-world buffer and the run dies with
   an illegal memory access. Default `nconmax`/`njmax` to `env.NCONMAX` /
   `env.NJMAX` when the caller passes none.
3. **Every constant goes in a dataclass**, never a literal in the reward body.
4. **Export your class** from `microdux/__init__.py`, in both the import and
   `__all__`, keeping the lists alphabetical.
5. **Do not touch files outside your task** except `rewards.py`, `curricula.py`,
   `sense.py`, `constants.py` and `__init__.py`, and only to add. Another agent
   is editing the same files in a different worktree, so additions merge and
   rewrites do not.

## Verifying

Write `tests/test_<task>.py` in the style of the existing tests, then smoke it:

```python
env = microdux.YourTask(envs=2)
state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.key(0), 2))
step = jax.jit(jax.vmap(env.step))
state = step(state, jnp.zeros((2, env.action_size)))
```

Assert the observation and reward are finite and the shapes are what the task
declares. Compiling one of these takes several minutes on CPU and the box is
memory-tight, so use `envs=2`, run one task per process, and never build two
models in the same interpreter.

Commit on your branch when it passes. Report what you ported, what you added to
the shared files, and anything upstream does that you could not reproduce.
