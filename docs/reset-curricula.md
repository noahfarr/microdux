# Curricula that move the start distribution do not work yet

Three ported tasks silently run upstream's step-0 spawn distribution for the
whole run: `Roulade` (standing/mid-roll mix), `RollerStandUp` (ground-state
mix) and, from earlier, `StandUp` and `SitStand` (also a ground-state mix).
Each was written by a different agent, and each reached the same conclusion
independently, which is why this is one gap rather than four compromises.

## Why

`reset(self, rng)` takes a key and nothing else. The training-step counter
lives in `info[PRESERVE]`, and `BraxAutoResetWrapper.step` restores it
*after* `reset` has already returned:

```python
next_info = jax.tree.map(where_done, reset_state.info, state.info)
...
next_info[preserve_info_key] = state.info[preserve_info_key]
```

So the counter survives across episode boundaries, which is what makes every
reward-weight curriculum work, since those read `elapsed` inside `step`. A
curriculum over the *initial state* has to run in `reset`, where the counter
is not visible.

## What upstream does instead

mjlab's `event_param_curriculum` reads a global step counter off the manager,
which is independent of any one environment. There is no equivalent here
because the env is a pure function of `rng`.

## The fix, when it is wanted

Give `reset` the counter. Two ways, both with real cost:

1. A `Respawn` wrapper outside the auto-reset, which re-samples the spawn
   with the restored counter wherever `done` fired. Correct and contained,
   but the resample traces on every step for every env, so it is paid whether
   or not anything reset.
2. Our own auto-reset in place of brax's, threading the counter into
   `reset`. No per-step cost, but it replaces a load-bearing wrapper on the
   path that produced the banked walking result.

Neither should land without re-running the walking task, since both touch the
reset path that the truncation fix already made delicate.

## Until then

Every affected task takes its mix as a constructor argument defaulting to
upstream's step-0 value, so a run can pick a fixed point on the schedule but
cannot move along it.

---

# Foot height is measured differently from upstream

`foot_clearance` and `foot_swing_height` read a terrain-height ray sensor
upstream: `RingPatternCfg.single_ring(radius=0.04, num_samples=2)` per foot
site, yaw aligned, parent body excluded. We use the foot site's own height
above the terrain instead, because MJX's `ray.py` has no heightfield support
and the analytic height is exact for a point and cheaper.

Measured on one frozen transition with every parameter matched (target height
0.02, command threshold 0.01, weight -2.0): `foot_clearance` comes out at
-0.002798 against upstream's -0.003199, a 13% difference on that term and
8e-6 on the total reward. Every other term of the flat velocity task agrees
to float32 rounding.

The difference is the measurement, not the formula. A tilted foot puts its
site at a different height from a 4 cm ring around it. Reproduce with
`tools/checkterms.py`.

---

# The roller velocity task converges to standing still

At 206M steps of 400M, `Rollers` travels 2 cm in eight seconds. It stands bolt
upright, tilt 0.0006, at 0.142 m with no crouch at all. Reward is flat from
113M to 206M, so it has settled rather than being on its way somewhere.

Per-term totals over a 400 step rollout explain it:

```
upright            799.0     42%
pose               796.1     42%
heading_hold       399.0     21%
wheel_speed         18.4      1%
glide, skating_air_time, gait_symmetry   0.0
single_support     -54.6            double support is penalised
```

Three posture terms that a motionless duck maximises trivially pay about
1994. Every skating-specific reward pays nothing. `wheel_speed` carries the
largest weight in the task, 10.0, and earns 1% of the total.

The reward is not unavailable. Commanded speed averages 0.173 m/s over the
rollout, so turning the wheels would pay roughly 760, a 37% gain. Claiming it
means disturbing the 2000 already banked, and `action_rate_l2` at -2.0 taxes
the exploration that would find it.

None of this is a porting difference. The weights match upstream's
RollerWeights, the velocity command range (-0.5, 0.6) matches
`command.ranges.lin_vel_x`, and `standing_fraction=0` matches
`rel_standing_envs`. Whether upstream's own training escapes this is unknown:
their logs ship one trained policy, the walking task, and no roller policy.

For contrast, `Swizzle` runs the same robot, the same roller physics and the
same `wheel_speed` term, and travels 34 cm in the same eight seconds.
Reproduce with `scratchpad/probe_breakdown.py`.
