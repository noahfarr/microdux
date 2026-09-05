import math

import jax.numpy as jnp

STEPS_PER_ITERATION = 24
UPSTREAM_ENVS = 4096


def iterations(count: int) -> int:
    return count * STEPS_PER_ITERATION * UPSTREAM_ENVS


def staircase(step, stages):
    thresholds = jnp.asarray([s[0] for s in stages])
    values = jnp.asarray([s[1] for s in stages])
    reached = jnp.sum(step >= thresholds) - 1
    return values[jnp.clip(reached, 0, len(stages) - 1)]


HEAD_POSE_RANGES = (
    (0, ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))),
    (iterations(500), ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))),
    (iterations(1000), ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))),
    (iterations(1500), ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.20, 0.20))),
    (iterations(2000), ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))),
)

HEAD_POSE_BIAS_WEIGHT = (
    (0, 0.0),
    (iterations(600), 1.0),
    (iterations(1000), 2.0),
    (iterations(1500), 3.0),
)

COM_RANGE = (
    (0, 0.003),
    (iterations(500), 0.005),
    (iterations(1000), 0.01),
    (iterations(1500), 0.015),
)

HEAD_COM_RANGE = (
    (0, 0.003),
    (iterations(500), 0.005),
    (iterations(1000), 0.01),
)

STANDING_FRACTION = (
    (0, 0.02),
    (iterations(500), 0.05),
    (iterations(750), 0.10),
    (iterations(1000), 0.15),
    (iterations(1500), 0.20),
    (iterations(2000), 0.25),
)

ACTION_RATE_WEIGHT = (
    (0, -0.1),
    (iterations(500), -0.2),
    (iterations(750), -0.4),
    (iterations(1000), -0.6),
    (iterations(1250), -0.8),
    (iterations(1500), -1.0),
)

FELL_OVER_LIMIT = (
    (0, math.radians(70.0)),
    (iterations(500), math.pi),
)

FALLEN_TAX_WEIGHT = (
    (0, 0.0),
    (iterations(1200), -0.5),
)

RECOVERY_SUCCESS_WEIGHT = (
    (0, 0.0),
    (iterations(1200), 10.0),
)

COM_UPWARD_WEIGHT = (
    (0, 0.0),
    (iterations(1200), 2.0),
)

GROUND_MIX = (
    (0, (0.40, 0.40, 0.20, 0.00)),
    (iterations(600), (0.25, 0.30, 0.35, 0.10)),
    (iterations(1500), (0.20, 0.25, 0.30, 0.25)),
    (iterations(2500), (0.15, 0.20, 0.30, 0.35)),
)


def head_ranges(step):
    thresholds = jnp.asarray([s[0] for s in HEAD_POSE_RANGES])
    values = jnp.asarray([s[1] for s in HEAD_POSE_RANGES])
    reached = jnp.clip(jnp.sum(step >= thresholds) - 1, 0, len(HEAD_POSE_RANGES) - 1)
    return values[reached]


def ranges(step, stages):
    thresholds = jnp.asarray([s[0] for s in stages])
    values = jnp.asarray([s[1] for s in stages])
    reached = jnp.clip(jnp.sum(step >= thresholds) - 1, 0, len(stages) - 1)
    return values[reached]


ROLLER_COM_RANGE = (
    (0, 0.003),
    (iterations(500), 0.005),
    (iterations(1000), 0.01),
)

SPIN_ACTION_RATE_WEIGHT = (
    (0, -0.5),
    (iterations(250), -0.8),
    (iterations(500), -1.0),
)

SPIN_LEG_ANTISYM_WEIGHT = (
    (0, 1.0),
    (iterations(1500), 0.5),
    (iterations(3000), 0.25),
)

SWIZZLE_ACTION_RATE_WEIGHT = (
    (0, -1.0),
    (iterations(250), -1.5),
    (iterations(500), -2.0),
)

SWIZZLE_HEADING_HOLD_WEIGHT = (
    (0, 1.0),
    (iterations(1000), 1.0),
    (iterations(1750), 0.5),
    (iterations(2500), 0.0),
)

SWIZZLE_HEADING_TRACKING_WEIGHT = (
    (0, 0.0),
    (iterations(1000), 0.0),
    (iterations(1750), 1.5),
    (iterations(2500), 3.0),
)

SWIZZLE_HEAD_POSE_TRACKING_WEIGHT = (
    (0, 0.0),
    (iterations(1500), 0.0),
    (iterations(2250), 2.0),
    (iterations(3000), 4.0),
)

SWIZZLE_HEAD_POSE_RANGES = (
    (0, ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))),
    (iterations(1500), ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))),
    (iterations(2250), ((-0.55, 0.55), (-0.55, 0.55), (-0.70, 0.70), (-0.15, 0.15))),
    (iterations(3000), ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))),
)

SITSTAND_RISE_SPEED_WEIGHT = (
    (0, 0.0),
    (iterations(1500), 5.0),
    (iterations(2500), 10.0),
)

ROLLER_ACTION_RATE_WEIGHT = (
    (0, -1.0),
    (iterations(250), -1.5),
    (iterations(500), -2.0),
)

GROUND_PICK_ACTION_RATE_WEIGHT = (
    (0, -0.8),
    (iterations(250), -1.5),
    (iterations(500), -2.0),
)

GROUND_PICK_COM_RANGE = (
    (0, 0.003),
    (iterations(500), 0.005),
    (iterations(1000), 0.01),
    (iterations(1500), 0.015),
    (iterations(2000), 0.02),
)

ROLLER_STANDUP_ACTION_RATE_WEIGHT = (
    (0, -0.4),
    (iterations(250), -0.8),
    (iterations(500), -1.0),
)

ROLLER_STANDUP_WHEEL_FRICTION = (
    (0, 0.0500),
    (iterations(1000), 0.0200),
    (iterations(2000), 0.0080),
    (iterations(3000), 0.0030),
    (iterations(4000), 0.0015),
)

ROLLER_STANDUP_PUSH_MAGNITUDE = (
    (0, 0.00),
    (iterations(500), 0.08),
    (iterations(1000), 0.20),
)

ROULADE_ACTION_RATE_WEIGHT = (
    (0, -0.1),
    (iterations(1500), -0.2),
    (iterations(3000), -0.4),
)

ROULADE_ARRIVAL_DAMPING_WEIGHT = (
    (0, 0.0),
    (iterations(2500), -0.025),
    (iterations(3500), -0.05),
)

ROULADE_TORQUE_RATE_WEIGHT = (
    (0, 0.0),
    (iterations(2500), -5e-4),
    (iterations(3500), -1e-3),
)

ROULADE_GENTLE_LANDING_WEIGHT = (
    (0, 0.002),
    (iterations(2500), 0.005),
)
