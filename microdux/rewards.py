import math

import jax
import jax.numpy as jnp
from flax import struct


def commanded(command, threshold):
    speed = jnp.linalg.norm(command[:2]) + jnp.abs(command[2])
    return (speed > threshold) * 1.0


def track_linear_velocity(velocity, command, std):
    error = jnp.sum(jnp.square(command[:2] - velocity[:2])) + jnp.square(velocity[2])
    return jnp.exp(-error / std**2)


def track_angular_velocity(rates, command, std):
    error = jnp.square(command[2] - rates[2]) + jnp.sum(jnp.square(rates[:2]))
    return jnp.exp(-error / std**2)


def upright(tilt, std):
    return jnp.exp(-tilt / std**2)


def variable_posture(joints, home, command, standing, walking, running,
                     walking_threshold, running_threshold):
    speed = jnp.linalg.norm(command[:2]) + jnp.abs(command[2])
    std = jnp.where(
        speed < walking_threshold,
        standing,
        jnp.where(speed < running_threshold, walking, running),
    )
    error = jnp.square(joints - home)
    return jnp.exp(-jnp.mean(error / std**2))


def body_angular_velocity(rates):
    return jnp.sum(jnp.square(rates[:2]))


def angular_momentum(angmom):
    return jnp.sum(jnp.square(angmom))


def joint_pos_limits(joints, limits):
    below = -jnp.clip(joints - limits[:, 0], max=0.0)
    above = jnp.clip(joints - limits[:, 1], min=0.0)
    return jnp.sum(below + above)


def action_rate(action, previous):
    return jnp.sum(jnp.square(action - previous))


def feet_air_time(air_time, command, threshold_min, threshold_max, command_threshold):
    live = (air_time > threshold_min) & (air_time < threshold_max)
    return jnp.sum(live * 1.0) * commanded(command, command_threshold)


def feet_clearance(height, velocity, command, target_height, command_threshold):
    speed = jnp.linalg.norm(velocity[:, :2], axis=-1)
    cost = jnp.sum(jnp.abs(height - target_height) * speed)
    return cost * commanded(command, command_threshold)


def feet_swing_height(peak, landed, command, target_height, command_threshold):
    error = peak / target_height - 1.0
    cost = jnp.sum(jnp.square(error) * landed * 1.0)
    return cost * commanded(command, command_threshold)


def feet_slip(velocity, contact, command, command_threshold):
    speed = jnp.linalg.norm(velocity[:, :2], axis=-1)
    cost = jnp.sum(jnp.square(speed) * contact * 1.0)
    return cost * commanded(command, command_threshold)


def self_collision(touching):
    return jnp.sum(touching * 1.0)


def head_pose_tracking(head, command, home, std):
    error = head - (home + command)
    return jnp.mean(jnp.exp(-jnp.square(error / std)))


def head_pose_bias(ema):
    return -jnp.mean(jnp.abs(ema))


def blend(previous, error, dt, tau):
    alpha = jnp.minimum(1.0, dt / jnp.maximum(tau, 1e-6))
    return (1.0 - alpha) * previous + alpha * error


@struct.dataclass
class Weights:
    track_linear_velocity: float = 2.0
    track_angular_velocity: float = 2.0
    upright: float = 2.0
    pose: float = 1.0
    body_ang_vel: float = -0.05
    angular_momentum: float = -0.02
    dof_pos_limits: float = -1.0
    action_rate_l2: float = -0.1
    air_time: float = 3.0
    foot_clearance: float = -2.0
    foot_swing_height: float = -0.25
    foot_slip: float = -0.1
    self_collisions: float = -1.0
    head_pose_tracking: float = 2.0
    head_pose_bias: float = 0.0


@struct.dataclass
class Tuning:
    linear_std: float = 0.1**0.5
    angular_std: float = 0.5**0.5
    upright_std: float = 0.05**0.5
    head_std: float = 0.5
    head_tau: float = 1.0
    walking_threshold: float = 0.01
    running_threshold: float = 1.5
    command_threshold: float = 0.01
    air_time_min: float = 0.125
    air_time_max: float = 0.300
    target_height: float = 0.02
    soft_limit_factor: float = 0.9


STANDING = {
    r".*hip_yaw.*": 0.1,
    r".*hip_roll.*": 0.05,
    r".*hip_pitch.*": 0.15,
    r".*knee.*": 0.15,
    r".*ankle.*": 0.1,
}

WALKING = {
    r".*hip_yaw.*": 0.3,
    r".*hip_roll.*": 0.05,
    r".*hip_pitch.*": 0.4,
    r".*knee.*": 0.4,
    r".*ankle.*": 0.25,
}

ROLLER_STANDING = {
    r".*hip_yaw.*": 0.05,
    r".*hip_roll.*": 0.05,
    r".*hip_pitch.*": 0.05,
    r".*knee.*": 0.05,
    r".*ankle.*": 0.05,
}

ROLLER_WALKING = {
    r".*hip_yaw.*": 0.3,
    r".*hip_roll.*": 0.6,
    r".*hip_pitch.*": 0.4,
    r".*knee.*": 0.4,
    r".*ankle.*": 0.25,
}

ROLLER_RUNNING = {
    r".*hip_yaw.*": 0.5,
    r".*hip_roll.*": 0.8,
    r".*hip_pitch.*": 0.8,
    r".*knee.*": 0.8,
    r".*ankle.*": 0.5,
}


def joint_deviation(joints, home):
    return jnp.sum(jnp.square(joints - home))


def joint_deviation_l1(joints, home):
    return jnp.sum(jnp.abs(joints - home))


def joint_torques(torque):
    return jnp.sum(jnp.square(torque))


def com_height_target(height, low, high):
    below = height < low
    above = height > high
    in_range = ~(below | above)
    penalty = jnp.square(height - low) * below + jnp.square(height - high) * above
    return in_range * 1.0 - penalty


def action_over_limit(target, limits, overshoot):
    over = jnp.clip(target - (limits[:, 1] + overshoot), min=0.0)
    under = jnp.clip((limits[:, 0] - overshoot) - target, min=0.0)
    return jnp.sum(over + under)


def hip_roll_neutral(joints, home):
    return jnp.sum(jnp.abs(joints - home))


def wheel_speed(cmd_x, omega, omega_scale, bidirectional):
    if bidirectional:
        aligned = jnp.sign(cmd_x) * omega
        return jnp.abs(cmd_x) * jnp.tanh(jnp.clip(aligned, min=0.0) / omega_scale)
    return jnp.clip(cmd_x, min=0.0) * jnp.tanh(jnp.clip(omega, min=0.0) / omega_scale)


def braking(cmd_x, forward_velocity, std):
    strength = jnp.clip(-cmd_x, min=0.0)
    stopped = jnp.exp(-jnp.square(jnp.clip(forward_velocity, min=0.0)) / std**2)
    return strength * stopped


def forward_gate(velocity, ref):
    return jnp.clip(jnp.clip(velocity, min=0.0) / ref, max=1.0)


def skating_air_time(air, threshold_min, threshold_max, cmd_x, gate):
    live = (air > threshold_min) & (air < threshold_max)
    return jnp.sum(live * 1.0) * jnp.clip(cmd_x, min=0.0) * gate


def single_support(contact_time, cmd_x, gate, double_penalty):
    n = jnp.sum((contact_time > 0.0) * 1.0)
    push = jnp.clip(cmd_x, min=0.0)
    single = (n == 1) * 1.0 * push * gate
    double = (n >= 2) * 1.0 * push * double_penalty
    return single - double


def glide(contact_time, joint_velocity, stillness_std, gate, cmd_x):
    n = jnp.sum((contact_time > 0.0) * 1.0)
    single = (n == 1) * 1.0
    stillness = jnp.exp(-jnp.sum(jnp.square(joint_velocity)) / stillness_std**2)
    active = (cmd_x >= 0.0) * 1.0
    return single * gate * stillness * active


def gait_symmetry(swing_accum):
    left, right = swing_accum
    return jnp.abs(left - right) / (left + right + 1e-3)


def leg_symmetry(left, right):
    return -jnp.mean(jnp.abs(left + right))


def grounded(n_contact, cmd_x):
    return (n_contact >= 2) * jnp.abs(cmd_x)


def heading_hold(yaw, reference, std):
    error = yaw - reference
    error = jnp.arctan2(jnp.sin(error), jnp.cos(error))
    return jnp.exp(-jnp.square(error) / std**2)


def heading_tracking(error, std):
    return jnp.exp(-jnp.square(error) / std**2)


def forward_lean(cmd_x, lean, target_pitch, std):
    push = jnp.clip(cmd_x, min=0.0)
    return push * jnp.exp(-jnp.square(lean - target_pitch) / std**2)


def feet_flat(xmat, gravity, contact=None):
    local = jnp.einsum("nji,j->ni", xmat, gravity)
    cost = jnp.sum(jnp.square(local[:, :2]), axis=-1)
    if contact is not None:
        cost = cost * contact
    return jnp.sum(cost)


def spin_envelope(phase, rate_max, accel_end, hold_end, brake_end):
    accel = rate_max * phase / accel_end
    brake = rate_max * (1.0 - (phase - hold_end) / (brake_end - hold_end))
    return jnp.where(
        phase < accel_end, accel,
        jnp.where(phase < hold_end, rate_max,
        jnp.where(phase < brake_end, brake, 0.0)),
    )


def spin_gate(phase, rate_max, accel_end, hold_end, brake_end):
    return spin_envelope(phase, rate_max, accel_end, hold_end, brake_end) / rate_max


def spin_rate_track(omega_z, target, std):
    return jnp.exp(-jnp.square((omega_z - target) / std))


def spin_rate_l1(omega_z, target):
    return -jnp.abs(omega_z - target)


def spin_stay_in_place(v_xy, scale):
    return jnp.sum(jnp.square(v_xy)) * scale


def spin_wheel_differential(diff, gate, omega_scale):
    return gate * jnp.tanh(jnp.clip(diff, min=0.0) / omega_scale)


def spin_grounded(n_contact, gate):
    return (n_contact >= 2) * gate


def leg_antisymmetry(left, right, gate):
    return gate * (-jnp.mean(jnp.abs(left - right)))


def crouch_blend(phase, descent_end, hold_end, rise_end):
    descend = phase / descent_end
    rise = 1.0 - (phase - hold_end) / (rise_end - hold_end)
    return jnp.where(
        phase < descent_end, descend,
        jnp.where(phase < hold_end, 1.0,
        jnp.where(phase < rise_end, rise, 0.0)),
    )


def forward_speed(velocity_x, ref):
    return jnp.tanh(jnp.clip(velocity_x, min=0.0) / ref)


def wheel_glide(omega_mean, radius, cap):
    return jnp.clip(omega_mean * radius, 0.0, cap)


def gate(value, low, high):
    return jnp.clip((value - low) / (high - low), 0.0, 1.0)


def pose_target(joints, target, std):
    return jnp.exp(-jnp.mean(jnp.square(joints - target)) / std**2)


def pose_l1(joints, target):
    return -jnp.mean(jnp.abs(joints - target))


def height_gaussian(height, target, std):
    return jnp.exp(-jnp.square(height - target) / std**2)


def height_l1(height, target):
    return -jnp.abs(height - target)


def upright_linear(cos_tilt):
    return cos_tilt


def vertical_accel(accel_z):
    return -jnp.abs(accel_z)


def downward_speed(velocity_z, cap):
    return -jnp.clip(-velocity_z - cap, 0.0, None)


def upward_speed(velocity_z, cap):
    return -jnp.clip(velocity_z - cap, 0.0, None)


def com_upward_velocity(velocity_z, height, max_height):
    return jnp.maximum(velocity_z, 0.0) * (height < max_height)


def smoothstep(x):
    t = jnp.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def upright_gate(height, cos_tilt, height_low, height_high, tilt_full_deg, tilt_zero_deg):
    reach = smoothstep((height - height_low) / max(height_high - height_low, 1e-6))
    tilt_deg = jnp.degrees(jnp.arccos(jnp.clip(cos_tilt, -1.0, 1.0)))
    upright = smoothstep((tilt_zero_deg - tilt_deg) / max(tilt_zero_deg - tilt_full_deg, 1e-6))
    return reach * upright


def gated(score, value, low, high):
    return score * gate(value, low, high)


def settled(height_error, tilt, velocity_norm, band_zero, band_full,
            tilt_zero, tilt_full, vel_std):
    quiet = jnp.exp(-jnp.square(velocity_norm / vel_std))
    return quiet * gate(height_error, band_zero, band_full) * gate(tilt, tilt_zero, tilt_full)


def composite(*scores):
    return jnp.prod(jnp.stack(scores))


@struct.dataclass
class StandUpWeights:
    pose_legs: float = 2.0
    pose_legs_l1: float = 1.25
    head_pose_tracking: float = 0.75
    head_pose_bias: float = 0.0
    height: float = 1.0
    height_sharp: float = 1.0
    height_l1: float = 7.5
    rise_bootstrap: float = 0.75
    gentle_rise: float = 0.005
    upright_linear: float = 1.5
    upright_sharp: float = 1.5
    standing_composite: float = 3.75
    body_ang_vel: float = -0.05
    angular_momentum: float = -0.02
    dof_pos_limits: float = -1.0
    action_rate_l2: float = -0.1
    self_collisions: float = -1.0


@struct.dataclass
class StandUpTuning:
    pose_std: float = 0.5
    height_std: float = 0.04
    height_sharp_std: float = 0.015
    upright_sharp_std: float = 0.3
    composite_height_std: float = 0.04
    composite_upright_std: float = 0.4
    composite_pose_std: float = 0.4
    rise_margin: float = 0.01
    head_std: float = 0.5
    head_tau: float = 1.0
    soft_limit_factor: float = 0.9


@struct.dataclass
class SitStandWeights:
    posture_pose_legs: float = 4.0
    posture_pose_l1: float = 1.0
    head_pose_tracking: float = 0.75
    posture_height: float = 1.0
    posture_height_sharp: float = 1.0
    posture_height_l1: float = 6.0
    rise_bootstrap: float = 0.75
    descent_speed: float = 10.0
    rise_speed: float = 0.0
    gentle_motion: float = 0.05
    upright_linear: float = 2.5
    upright_while_tall: float = 1.5
    posture_stillness: float = 2.0
    posture_composite: float = 3.0
    body_ang_vel: float = -0.05
    angular_momentum: float = -0.02
    dof_pos_limits: float = -1.0
    action_rate_l2: float = -0.1
    self_collisions: float = -1.0


@struct.dataclass
class SitStandTuning:
    pose_std: float = 0.5
    height_std: float = 0.04
    height_sharp_std: float = 0.015
    rise_margin: float = 0.01
    max_descent_speed: float = 0.05
    max_rise_speed: float = 0.08
    stand_upright_z: float = 0.10
    sit_upright_z: float = 0.075
    stillness_band_full: float = 0.012
    stillness_band_zero: float = 0.03
    stillness_vel_std: float = 0.05
    stillness_tilt_full_deg: float = 25.0
    stillness_tilt_zero_deg: float = 60.0
    composite_height_std: float = 0.03
    composite_upright_std: float = 0.40
    composite_pose_std: float = 0.40
    composite_head_std: float = 0.40
    head_std: float = 0.5
    head_tau: float = 1.0
    soft_limit_factor: float = 0.9


@struct.dataclass
class RollerWeights:
    pose: float = 2.0
    upright: float = 2.0
    body_ang_vel: float = -0.05
    angular_momentum: float = -0.02
    action_rate_l2: float = -1.0
    com_height_target: float = 2.0
    self_collisions: float = -1.0
    feet_flat: float = -2.0
    neck_action_rate_l2: float = -0.5
    neck_joint_pos_l2: float = -0.5
    joint_torques_l2: float = -1e-3
    action_over_limit: float = -0.5
    hip_roll_neutral: float = -2.0
    wheel_speed: float = 10.0
    braking: float = 1.0
    skating_air_time: float = 1.5
    glide: float = 4.0
    single_support: float = 3.0
    gait_symmetry: float = -1.0
    forward_lean: float = 1.5
    heading_hold: float = 1.0


@struct.dataclass
class RollerStandUpWeights:
    pose_legs: float = 8.0
    pose_legs_l1: float = 5.0
    height: float = 4.0
    height_sharp: float = 4.0
    height_l1: float = 30.0
    rise_bootstrap: float = 3.0
    gentle_rise: float = 0.02
    upright_linear: float = 6.0
    upright_sharp: float = 6.0
    standing_composite: float = 15.0
    joint_torque_rate_l2: float = -0.2
    body_ang_vel: float = -0.05
    angular_momentum: float = -0.02
    action_rate_l2: float = -0.6
    self_collisions: float = -1.0
    neck_action_rate_l2: float = -0.5
    neck_joint_pos_l2: float = -0.5
    joint_torques_l2: float = -1e-3
    action_over_limit: float = -0.5


@struct.dataclass
class RollerStandUpTuning:
    pose_std: float = 0.5
    height_std: float = 0.04
    height_sharp_std: float = 0.015
    upright_sharp_std: float = 0.3
    composite_height_std: float = 0.04
    composite_upright_std: float = 0.40
    composite_pose_std: float = 0.40
    rise_margin: float = 0.010
    action_overshoot: float = 0.3
    stand_z: float = 0.138
    prone_z: float = 0.075


@struct.dataclass
class RollerTuning:
    upright_std: float = 0.05**0.5
    walking_threshold: float = 0.01
    running_threshold: float = 0.5
    soft_limit_factor: float = 0.9
    com_height_min: float = 0.0935
    com_height_max: float = 0.1235
    action_overshoot: float = 0.3
    wheel_radius: float = 0.0175
    wheel_vel_scale: float = 0.3
    braking_std: float = 0.3
    skating_air_min: float = 0.15
    skating_air_max: float = 0.45
    skating_vel_gate: float = 0.2
    single_support_double_penalty: float = 0.25
    single_support_vel_gate: float = 0.2
    glide_stillness_std: float = 5.0
    glide_vel_ref: float = 0.2
    forward_lean_target: float = 0.262
    forward_lean_std: float = 0.1
    heading_hold_std: float = 0.4


def ball_forward_velocity(velocity_xy, direction, max_speed):
    forward = jnp.dot(velocity_xy, direction)
    return jnp.clip(jnp.nan_to_num(forward), 0.0, max_speed)


def ball_speed_overshoot(velocity_xy, direction, target_speed, max_penalty):
    forward = jnp.dot(velocity_xy, direction)
    over = jnp.nan_to_num(forward) - target_speed
    return jnp.clip(over, 0.0, max_penalty)


def feet_grounded(touching):
    count = touching.shape[0]
    return jnp.clip(jnp.sum(touching * 1.0), 0.0, float(count)) / float(count)


def body_impact(force, threshold):
    return jnp.clip(jnp.linalg.norm(force) - threshold, min=0.0)


def phase_descend(phase, descent_end, hold_end, rise_end):
    down = phase / jnp.maximum(descent_end, 1e-9)
    rise = 1.0 - (phase - hold_end) / jnp.maximum(rise_end - hold_end, 1e-9)
    return jnp.where(
        phase < descent_end, down,
        jnp.where(phase < hold_end, 1.0, jnp.where(phase < rise_end, rise, 0.0)),
    )


def phase_rise(phase, hold_end, rise_end):
    rising = (phase - hold_end) / jnp.maximum(rise_end - hold_end, 1e-9)
    return jnp.where(phase < hold_end, 0.0, jnp.where(phase < rise_end, rising, 1.0))


def mouth_alignment(xmat):
    return -xmat[2, 0]


def joint_vel_l2(vel):
    return jnp.mean(jnp.square(vel))


@struct.dataclass
class BallKickWeights:
    ball_forward_velocity: float = 12.0
    ball_speed_overshoot: float = -4.0
    support_foot_grounded: float = 2.0
    pose_stand_legs: float = 2.0
    pose_stand_neck: float = 1.0
    upright: float = 2.0
    height_stand: float = 1.0
    body_ang_vel: float = -0.05
    angular_momentum: float = -0.02
    dof_pos_limits: float = -1.0
    action_rate_l2: float = -0.1
    self_collisions: float = -1.0


@struct.dataclass
class BallKickTuning:
    pose_std: float = 0.5
    neck_std: float = 0.3
    upright_std: float = 0.05**0.5
    height_std: float = 0.04
    stand_height: float = 0.115
    target_speed: float = 1.0
    overshoot_cap: float = 5.0
    soft_limit_factor: float = 0.9
    standing_z: tuple = (0.11, 0.12)
    standing_tilt_deg: float = 5.0
    joint_noise: float = 0.05
    ball_radius: float = 0.035
    offset_x: float = 0.09
    offset_abs_y: float = 0.042
    offset_noise: float = 0.015


@struct.dataclass
class GroundPickWeights:
    mouth_ground_proximity: float = 3.0
    mouth_perpendicular: float = 2.0
    return_pose_legs: float = 6.0
    return_pose_neck: float = 6.0
    return_upright: float = 4.0
    neck_vel_descent: float = -0.1
    upright: float = 0.2
    body_ang_vel: float = -0.05
    angular_momentum: float = -0.02
    dof_pos_limits: float = -1.0
    feet_grounded: float = 3.0
    feet_flat: float = -2.0
    action_rate_l2: float = -0.8
    neck_action_rate_l2: float = -1.0
    joint_torques_l2: float = -5e-3
    self_collisions: float = -1.0
    head_impact_penalty: float = -2.0


@struct.dataclass
class GroundPickTuning:
    mouth_std: float = 0.10
    return_pose_std: float = 0.3
    return_neck_std: float = 0.15
    return_upright_std: float = 0.4
    upright_std: float = 0.05**0.5
    head_impact_threshold: float = 1.0
    soft_limit_factor: float = 0.9
    period: float = 4.0
    descent_end: float = 0.375
    hold_end: float = 0.425
    rise_end: float = 0.80
    payload_min_kg: float = 0.01
    payload_max_kg: float = 0.04
    payload_ramp: float = 0.05
    gravity: float = 9.81
def progress_rate(delta, dt, target):
    return delta / (dt * target)


def rate_overspeed(rate, cap):
    return jnp.square(jnp.clip(jnp.abs(rate) - cap, min=0.0))


def head_pivot_score(contact, window, rate, top_down):
    return contact * window * jnp.clip(rate, 0.0, 1.0) * (0.3 + 0.7 * top_down)


def upright_bootstrap(cos_tilt):
    return jnp.clip(cos_tilt, min=0.0)


def height_shortfall(height, target):
    return -jnp.clip(target - height, min=0.0)


def off_axis_angular_velocity(omega_body):
    return jnp.square(omega_body[0]) + jnp.square(omega_body[2])


def component_sq(vector, index):
    return jnp.square(vector[index])


@struct.dataclass
class RouladeWeights:
    progress: float = 8.0
    overspeed: float = -0.1
    head_pivot: float = 0.5
    landing_composite: float = 4.0
    upright_after_roll: float = 1.5
    height_after_roll: float = 1.0
    landing_sharp: float = 2.0
    stand_tax: float = 5.0
    rise_velocity: float = 0.75
    sagittal: float = -0.1
    lateral_vel: float = -0.5
    flatness: float = -0.5
    action_rate_l2: float = -0.1
    torque_rate_l2: float = 0.0
    body_ang_vel: float = -0.002
    angular_momentum: float = -0.001
    dof_pos_limits: float = -1.0
    arrival_damping: float = 0.0
    gentle_landing: float = 0.002
    self_collisions: float = -0.1
    head_pose_tracking: float = 2.0
    head_pose_bias: float = 0.0


@struct.dataclass
class RouladeTuning:
    target_angle: float = 2 * math.pi
    max_paid_rate: float = 5.0
    omega_max: float = 7.0
    head_pivot_lo: float = math.radians(30.0)
    head_pivot_hi: float = math.radians(240.0)
    head_pivot_rate_norm: float = 2.0
    landing_height_std: float = 0.04
    landing_upright_std: float = 0.40
    landing_pose_std: float = 0.40
    sharp_height_std: float = 0.015
    sharp_upright_std: float = 0.3
    height_after_roll_std: float = 0.04
    rise_margin: float = 0.01
    gate_lo: float = math.radians(260.0)
    gate_hi: float = math.radians(330.0)
    rise_gate_lo: float = math.radians(180.0)
    rise_gate_hi: float = math.radians(260.0)
    arrival_height_low: float = 0.09
    arrival_height_high: float = 0.11
    arrival_tilt_full_deg: float = 20.0
    arrival_tilt_zero_deg: float = 45.0
    head_std: float = 0.5
    head_tau: float = 1.0
    soft_limit_factor: float = 0.9
