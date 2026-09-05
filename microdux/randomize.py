import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

HEAD_BODIES = ("neck", "neck_pitch", "yaw_roll_motion", "bottom_head_shell", "jaw_soft", "bearing_roll")


@struct.dataclass
class Spec:
    com: float = 0.003
    head_com: float = 0.003
    mass_inertia: tuple = (0.95, 1.05)
    armature: tuple = (0.9, 1.1)
    joint_friction: tuple = (0.9, 1.1)
    foot_friction: tuple = (0.7, 1.3)
    vin: tuple = (6.5, 8.2)
    vin_drop_gain: tuple = (0.0, 0.2)
    encoder_bias: tuple = (-0.015, 0.015)
    imu_angle: float = 6.0
    push_interval: tuple = (3.0, 6.0)
    push: tuple = (-0.3, 0.3)
    base_pitch: float = 0.0
    base_roll: float = 0.0
    base_height: tuple = (0.12, 0.13)
    entry_velocity: tuple = (0.0, 0.0)
    kp: tuple | None = None
    kd: tuple | None = None


@struct.dataclass
class Draw:
    trunk_com: jax.Array
    head_com: jax.Array
    body_mass: jax.Array
    body_inertia: jax.Array
    dof_armature: jax.Array
    geom_friction: jax.Array
    encoder_bias: jax.Array
    imu_quat: jax.Array
    push_timer: jax.Array


@struct.dataclass
class Slots:
    trunk: np.ndarray = struct.field(pytree_node=False)
    head: np.ndarray = struct.field(pytree_node=False)
    dofs: np.ndarray = struct.field(pytree_node=False)
    feet: np.ndarray = struct.field(pytree_node=False)
    nu: int = struct.field(pytree_node=False)


def slots(mj_model, layout) -> Slots:
    import mujoco

    def body(name):
        return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)

    head = np.array([i for i in (body(n) for n in HEAD_BODIES) if i >= 0])
    return Slots(
        trunk=np.array([layout.trunk_body]),
        head=head,
        dofs=np.asarray(layout.actuated_qvel_adr),
        feet=np.asarray(layout.foot_geoms),
        nu=int(mj_model.nu),
    )


def axis_angle(key, degrees):
    direction_key, angle_key = jax.random.split(key)
    axis = jax.random.normal(direction_key, (3,))
    axis = axis / jnp.maximum(jnp.linalg.norm(axis), 1e-9)
    angle = jax.random.uniform(angle_key, minval=0.0, maxval=jnp.deg2rad(degrees))
    return jnp.concatenate([jnp.cos(angle / 2)[None], axis * jnp.sin(angle / 2)])


def draw(key, mjx_model, spec: Spec, place: Slots, dt: float) -> Draw:
    keys = jax.random.split(key, 9)

    trunk_com = jax.random.uniform(keys[0], (place.trunk.size, 3), minval=-1.0, maxval=1.0)
    head_com = jax.random.uniform(keys[1], (max(place.head.size, 1), 3), minval=-1.0, maxval=1.0)

    scale = jax.random.uniform(keys[2], minval=spec.mass_inertia[0], maxval=spec.mass_inertia[1])

    armature = mjx_model.dof_armature
    armature = armature.at[place.dofs].multiply(
        jax.random.uniform(keys[3], (place.dofs.size,),
                           minval=spec.armature[0], maxval=spec.armature[1])
    )

    friction = mjx_model.geom_friction
    if place.feet.size:
        friction = friction.at[place.feet, 0].set(
            jax.random.uniform(keys[4], (place.feet.size,),
                               minval=spec.foot_friction[0], maxval=spec.foot_friction[1])
        )

    return Draw(
        trunk_com=trunk_com,
        head_com=head_com,
        body_mass=mjx_model.body_mass * scale,
        body_inertia=mjx_model.body_inertia * scale,
        dof_armature=armature,
        geom_friction=friction,
        encoder_bias=jax.random.uniform(
            keys[5], (place.nu,), minval=spec.encoder_bias[0], maxval=spec.encoder_bias[1]),
        imu_quat=axis_angle(keys[6], spec.imu_angle),
        push_timer=interval(keys[7], spec.push_interval, dt),
    )


def interval(key, span, dt):
    seconds = jax.random.uniform(key, minval=span[0], maxval=span[1])
    return jnp.ceil(seconds / dt).astype(jnp.int32)


def servos(key, spec: Spec, nu: int):
    from . import plant

    vin_key, sag_key, friction_key = jax.random.split(key, 3)
    return plant.Servos(
        vin=jax.random.uniform(vin_key, (1,), minval=spec.vin[0], maxval=spec.vin[1]),
        vin_drop_gain=jax.random.uniform(
            sag_key, (1,), minval=spec.vin_drop_gain[0], maxval=spec.vin_drop_gain[1]),
        kp_scale=jnp.ones(1),
        kd_scale=jnp.ones(1),
        friction_scale=jax.random.uniform(
            friction_key, (1,),
            minval=spec.joint_friction[0], maxval=spec.joint_friction[1]),
        previous_motor_torque=jnp.zeros(nu),
    )


def apply(mjx_model, sample: Draw, place: Slots, com: float, head_com: float):
    ipos = mjx_model.body_ipos.at[place.trunk].add(sample.trunk_com * com)
    if place.head.size:
        ipos = ipos.at[place.head].add(sample.head_com * head_com)

    return mjx_model.tree_replace({
        "body_ipos": ipos,
        "body_mass": sample.body_mass,
        "body_inertia": sample.body_inertia,
        "dof_armature": sample.dof_armature,
        "geom_friction": sample.geom_friction,
    })


def tilted(key, spec: Spec, height_span: tuple, pitch_range: tuple | None = None,
           roll_range: tuple | None = None):
    pitch_key, roll_key, height_key = jax.random.split(key, 3)
    pitch_range = pitch_range or (-spec.base_pitch, spec.base_pitch)
    roll_range = roll_range or (-spec.base_roll, spec.base_roll)
    pitch = jax.random.uniform(
        pitch_key, minval=jnp.deg2rad(pitch_range[0]), maxval=jnp.deg2rad(pitch_range[1]))
    roll = jax.random.uniform(
        roll_key, minval=jnp.deg2rad(roll_range[0]), maxval=jnp.deg2rad(roll_range[1]))
    height = jax.random.uniform(height_key, minval=height_span[0], maxval=height_span[1])

    half_pitch, half_roll = pitch / 2.0, roll / 2.0
    quat = jnp.array([
        jnp.cos(half_roll) * jnp.cos(half_pitch),
        jnp.sin(half_roll) * jnp.cos(half_pitch),
        jnp.cos(half_roll) * jnp.sin(half_pitch),
        -jnp.sin(half_roll) * jnp.sin(half_pitch),
    ])
    return quat, height


def shove(key, spec: Spec):
    return jax.random.uniform(key, (2,), minval=spec.push[0], maxval=spec.push[1])


@struct.dataclass
class GroundMix:
    standing: float = 0.25
    sitting: float = 0.25
    face_down: float = 0.25
    face_up: float = 0.25
    standing_height: tuple = (0.11, 0.12)
    sitting_height: tuple = (0.05, 0.09)
    prone_height: tuple = (0.05, 0.09)
    sitting_tilt: float = 10.0
    face_up_roll: float = 90.0
    sitting_noise: float = 0.12


def groundstate(key, mix: GroundMix, home_joints, sit_joints):
    category_key, pose_key, noise_key = jax.random.split(key, 3)
    weights = jnp.array([mix.standing, mix.sitting, mix.face_down, mix.face_up])
    category = jax.random.categorical(category_key, jnp.log(jnp.maximum(weights, 1e-9)))

    dummy = Spec()
    standing_quat, standing_z = tilted(
        pose_key, dummy, mix.standing_height, (-3.0, 3.0), (-3.0, 3.0))
    sitting_quat, sitting_z = tilted(
        pose_key, dummy, mix.sitting_height, (-mix.sitting_tilt, mix.sitting_tilt),
        (-mix.sitting_tilt, mix.sitting_tilt))
    down_quat, down_z = tilted(pose_key, dummy, mix.prone_height, (80.0, 100.0), (-10.0, 10.0))
    up_quat, up_z = tilted(
        pose_key, dummy, mix.prone_height, (-100.0, -80.0), (-mix.face_up_roll, mix.face_up_roll))

    quats = jnp.stack([standing_quat, sitting_quat, down_quat, up_quat])
    heights = jnp.stack([standing_z, sitting_z, down_z, up_z])

    noise = jax.random.uniform(
        noise_key, home_joints.shape, minval=-mix.sitting_noise, maxval=mix.sitting_noise)
    joints = jnp.where(category == 1, sit_joints + noise, home_joints)

    return joints, quats[category], heights[category]
