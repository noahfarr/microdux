import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct

GRAVITY = jnp.array([0.0, 0.0, -1.0])


def address(mj_model, name: str):
    index = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if index < 0:
        raise KeyError(f"model has no sensor {name!r}")
    start = int(mj_model.sensor_adr[index])
    return start, start + int(mj_model.sensor_dim[index])


@struct.dataclass
class Sensors:
    ang_vel: tuple = struct.field(pytree_node=False)
    lin_vel: tuple = struct.field(pytree_node=False)
    accel: tuple = struct.field(pytree_node=False)
    angmom: tuple = struct.field(pytree_node=False)
    orientation: tuple = struct.field(pytree_node=False)
    feet_linvel: tuple = struct.field(pytree_node=False)


def sensors(mj_model, prefix: str = "") -> Sensors:
    return Sensors(
        ang_vel=address(mj_model, prefix + "imu_ang_vel"),
        lin_vel=address(mj_model, prefix + "imu_lin_vel"),
        accel=address(mj_model, prefix + "imu_accel"),
        angmom=address(mj_model, prefix + "root_angmom"),
        orientation=address(mj_model, prefix + "orientation"),
        feet_linvel=tuple(
            address(mj_model, f"{prefix}{site}_linvel")
            for site in ("left_foot", "right_foot")
        ),
    )


def read(data, span):
    return data.sensordata[span[0]:span[1]]


def rotate_inverse(quat, vector):
    w, x, y, z = quat
    conjugate = jnp.array([w, -x, -y, -z])
    return rotate(conjugate, vector)


def rotate(quat, vector):
    w, xyz = quat[0], quat[1:]
    t = 2.0 * jnp.cross(xyz, vector)
    return vector + w * t + jnp.cross(xyz, t)


def gravity(data, spans: Sensors):
    return rotate_inverse(read(data, spans.orientation), GRAVITY)


def tilt(data, spans: Sensors):
    return jnp.sum(jnp.square(gravity(data, spans)[:2]))


def cos_tilt(data, spans: Sensors):
    return -gravity(data, spans)[2]


def soft_limits(mj_model, layout, factor: float = 0.9):
    ranges = mj_model.jnt_range[layout.actuated_joint_ids]
    middle = ranges.mean(axis=1, keepdims=True)
    return middle + (ranges - middle) * factor


def foot_velocity(data, spans: Sensors):
    return jnp.stack([read(data, span) for span in spans.feet_linvel])


def root_quat(data, base: int = 0):
    return data.qpos[base + 3:base + 7]


def root_linear_velocity(data, base: int = 0, dof: int = 0):
    return rotate_inverse(root_quat(data, base), data.qvel[dof:dof + 3])


def root_angular_velocity(data, dof: int = 0):
    return data.qvel[dof + 3:dof + 6]


def world_angular_velocity(data, base: int = 0, dof: int = 0):
    return rotate(root_quat(data, base), root_angular_velocity(data, dof))


def yaw(quat):
    w, x, y, z = quat
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def euler_zyx(yaw, pitch, roll):
    cy, sy = jnp.cos(yaw * 0.5), jnp.sin(yaw * 0.5)
    cp, sp = jnp.cos(pitch * 0.5), jnp.sin(pitch * 0.5)
    cr, sr = jnp.cos(roll * 0.5), jnp.sin(roll * 0.5)
    return jnp.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
