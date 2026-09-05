import functools
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import actuator as bam_actuator
from . import constants
from . import terrain as rough

SCENES = {
    "walk": "scene_walk.xml",
    "allcollisions": "scene.xml",
    "rollers": "scene_rollers.xml",
    "walk_backlash": "scene_walk_backlash.xml",
    "backlash": "scene_backlash.xml",
    "rollers_backlash": "scene_rollers_backlash.xml",
}

HULLS = {
    "sole_left": "sole_left_hull",
    "sole_right": "sole_right_hull",
    "leg": "leg_hull",
    "power_support": "power_support_hull",
}

WHEEL_BODIES = ("tire", "tire_2", "tire_3", "tire_4")
WHEEL_RADIUS = 0.015
WHEEL_HALF_WIDTH = 0.0038
WHEEL_AXIS_FIX = (0.7071067811865476, 0.0, 0.7071067811865475, 0.0)


@dataclass
class Layout:
    joints: list[str]
    actuators: list[str]
    actuated_joint_ids: np.ndarray
    actuated_qpos_adr: np.ndarray
    actuated_qvel_adr: np.ndarray
    home: np.ndarray
    trunk_body: int
    imu_site: int
    foot_sites: np.ndarray
    foot_geoms: np.ndarray
    floor_geom: int
    friction_dofs: np.ndarray
    backlash_qpos_adr: np.ndarray
    backlash_qvel_adr: np.ndarray
    backlash_mask: np.ndarray
    keyframes: dict[str, int] = field(default_factory=dict)


@functools.lru_cache(maxsize=None)
def _alignments(path, wanted) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    probe = mujoco.MjSpec.from_file(str(path))
    probe.compiler.meshdir = "assets"
    named = {m.name for m in probe.meshes}
    for hull in sorted(wanted):
        if hull not in named:
            probe.add_mesh(name=hull, file=f"{hull}.stl")
    compiled = probe.compile()

    frames = {}
    for source in list(HULLS) + list(HULLS.values()):
        index = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_MESH, source)
        if index >= 0:
            frames[source] = (
                np.asarray(compiled.mesh_pos[index], dtype=float),
                np.asarray(compiled.mesh_quat[index], dtype=float),
            )
    return frames


def _compose(pose, other):
    position, rotation = pose
    offset, turn = other
    moved = np.zeros(3)
    mujoco.mju_rotVecQuat(moved, offset, rotation)
    combined = np.zeros(4)
    mujoco.mju_mulQuat(combined, rotation, turn)
    return position + moved, combined


def _invert(pose):
    position, rotation = pose
    inverse = np.zeros(4)
    mujoco.mju_negQuat(inverse, rotation)
    moved = np.zeros(3)
    mujoco.mju_rotVecQuat(moved, -position, inverse)
    return moved, inverse


def _collides(geom) -> bool:
    return bool(geom.contype) or bool(geom.conaffinity)


def _swap_hulls(spec: mujoco.MjSpec, path) -> int:
    wanted = {
        HULLS[g.meshname] for g in spec.geoms if g.meshname in HULLS and _collides(g)
    }
    if not wanted:
        return 0

    frames = _alignments(str(path), frozenset(wanted))
    named = {m.name for m in spec.meshes}
    for hull in sorted(wanted):
        if hull not in named:
            spec.add_mesh(name=hull, file=f"{hull}.stl")

    swapped = 0
    for geom in spec.geoms:
        source = geom.meshname
        if source not in HULLS or not _collides(geom):
            continue
        hull = HULLS[source]
        pose = (np.asarray(geom.pos, dtype=float), np.asarray(geom.quat, dtype=float))
        pose = _compose(pose, frames[source])
        pose = _compose(pose, _invert(frames[hull]))

        geom.pos, geom.quat = pose
        geom.meshname = hull
        swapped += 1
    return swapped


@functools.lru_cache(maxsize=None)
def _wheel_frame(path):
    probe = mujoco.MjModel.from_xml_path(str(path))
    body = mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_BODY, WHEEL_BODIES[0])
    if body < 0:
        return None
    start, count = probe.body_geomadr[body], probe.body_geomnum[body]
    colliding = [
        g for g in range(start, start + count)
        if probe.geom_contype[g] or probe.geom_conaffinity[g]
    ]
    if not colliding:
        return None
    g = colliding[0]
    return np.asarray(probe.geom_pos[g], dtype=float), np.asarray(probe.geom_quat[g], dtype=float)


WHEEL_GROUP = 4


def _swap_wheels(spec: mujoco.MjSpec, path) -> int:
    frame = _wheel_frame(str(path))
    if frame is None:
        return 0

    swapped = 0
    for body in spec.bodies:
        if body.name not in WHEEL_BODIES:
            continue
        for geom in body.geoms:
            if not _collides(geom):
                continue
            geom.pos, geom.quat = _compose(frame, (np.zeros(3), WHEEL_AXIS_FIX))
            geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            geom.size = [WHEEL_RADIUS, WHEEL_HALF_WIDTH, 0.0]
            geom.meshname = ""
            geom.contype = WHEEL_GROUP
            geom.conaffinity = WHEEL_GROUP
            swapped += 1

    if swapped:
        for geom in spec.geoms:
            if geom.name == "floor":
                geom.contype |= WHEEL_GROUP
                geom.conaffinity |= WHEEL_GROUP
    return swapped


STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)


def _to_bam_actuators(
    spec: mujoco.MjSpec, bam: bam_actuator.Bam, vin_max: float, stiff: bool = True
) -> int:
    force_limit = vin_max * bam.kt / bam.resistance
    targets = {a.target for a in spec.actuators}
    converted = 0

    for act in spec.actuators:
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.gear = [1.0, 0, 0, 0, 0, 0]
        converted += 1

    for joint in spec.joints:
        if joint.name in targets:
            joint.armature = float(bam.armature)
            joint.damping = np.zeros((3, 1))
            joint.frictionloss = float(bam.friction_base)
            if stiff:
                joint.solref_friction = STIFF_SOLREF_FRICTION
                joint.solimp_friction = STIFF_SOLIMP_FRICTION

    return converted


def _add_foot_sensors(spec: mujoco.MjSpec) -> int:
    sites = {s.name for s in spec.sites}
    present = {s.name for s in spec.sensors}
    added = 0
    for site in constants.FEET_SITES:
        if site not in sites:
            continue
        name = f"{site}_linvel"
        if name in present:
            continue
        sensor = spec.add_sensor()
        sensor.name = name
        sensor.type = mujoco.mjtSensor.mjSENS_FRAMELINVEL
        sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
        sensor.objname = site
        added += 1
    return added


def build(
    variant: str = "walk",
    bam: bam_actuator.Bam | None = None,
    vin_max: float = 8.2,
    timestep: float = 0.005,
    iterations: int = 10,
    ls_iterations: int = 20,
    self_collision: bool = True,
    hulls: bool = True,
    stiff: bool = True,
    terrain: rough.Terrain | None = None,
) -> tuple[mujoco.MjModel, Layout]:
    if variant not in SCENES:
        raise KeyError(f"unknown variant {variant!r}, have {sorted(SCENES)}")
    bam = bam or bam_actuator.load(kp=200.0)

    path = constants.XMLS / SCENES[variant]
    spec = mujoco.MjSpec.from_file(str(path))
    spec.compiler.meshdir = "assets"

    if hulls:
        _swap_hulls(spec, path)
        _swap_wheels(spec, path)
    _to_bam_actuators(spec, bam, vin_max, stiff)
    _add_foot_sensors(spec)

    if self_collision:
        group = spec.find_default("self_collision_only")
        if group is not None:
            group.geom.contype = 2
            group.geom.conaffinity = 2

    if terrain is not None:
        rough.attach(spec, terrain)

    spec.option.timestep = timestep
    spec.option.iterations = iterations
    spec.option.ls_iterations = ls_iterations

    model = spec.compile()
    return model, layout(model)


def _backlash(model, actuators, qpos_adr, qvel_adr):
    positions, velocities, mask = [], [], []
    for name, qp, qv in zip(actuators, qpos_adr, qvel_adr):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"passive_{name}_backlash")
        if joint >= 0:
            positions.append(model.jnt_qposadr[joint])
            velocities.append(model.jnt_dofadr[joint])
            mask.append(1.0)
        else:
            positions.append(qp)
            velocities.append(qv)
            mask.append(0.0)
    return np.array(positions), np.array(velocities), np.array(mask)


def lone(bam: bam_actuator.Bam, vin_max: float = 8.2, robot=None, stiff: bool = True,
         self_collision: bool = True) -> mujoco.MjSpec:
    robot = robot or constants.WALK
    duck = mujoco.MjSpec.from_file(str(robot))
    duck.compiler.meshdir = "assets"
    _swap_hulls(duck, robot)
    _to_bam_actuators(duck, bam, vin_max, stiff)
    _add_foot_sensors(duck)
    if self_collision:
        group = duck.find_default("self_collision_only")
        if group is not None:
            group.geom.contype = 2
            group.geom.conaffinity = 2
    return duck


def layout(model: mujoco.MjModel) -> Layout:
    def named(kind, count):
        return [mujoco.mj_id2name(model, kind, i) or "" for i in range(count)]

    def ids(kind, names):
        found = [mujoco.mj_name2id(model, kind, n) for n in names]
        return np.array([i for i in found if i >= 0], dtype=int)

    joints = named(mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    actuators = named(mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    geoms = named(mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)

    joint_ids = np.array([joints.index(a) for a in actuators])
    qpos_adr = model.jnt_qposadr[joint_ids]
    qvel_adr = model.jnt_dofadr[joint_ids]
    backlash_qpos_adr, backlash_qvel_adr, backlash_mask = _backlash(
        model, actuators, qpos_adr, qvel_adr
    )

    return Layout(
        joints=joints,
        actuators=actuators,
        actuated_joint_ids=joint_ids,
        actuated_qpos_adr=qpos_adr,
        actuated_qvel_adr=qvel_adr,
        home=np.asarray(constants.resolve(constants.HOME, actuators)),
        trunk_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, constants.TRUNK),
        imu_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, constants.IMU_SITE),
        foot_sites=ids(mujoco.mjtObj.mjOBJ_SITE, constants.FEET_SITES),
        foot_geoms=ids(mujoco.mjtObj.mjOBJ_GEOM, constants.FOOT_GEOMS),
        floor_geom=geoms.index("floor") if "floor" in geoms else -1,
        friction_dofs=np.nonzero(model.dof_frictionloss > 0)[0],
        backlash_qpos_adr=backlash_qpos_adr,
        backlash_qvel_adr=backlash_qvel_adr,
        backlash_mask=backlash_mask,
        keyframes={name: i for i, name in enumerate(named(mujoco.mjtObj.mjOBJ_KEY, model.nkey))},
    )


def to_mjx(mj_model: mujoco.MjModel, impl: str = "jax"):
    from mujoco import mjx

    if impl != "warp":
        return mjx.put_model(mj_model, impl=impl)

    try:
        from warp._src.jax.ffi import GraphMode
    except ImportError:
        from warp.jax_experimental.ffi import GraphMode

    return mjx.put_model(mj_model, impl=impl, graph_mode=GraphMode.WARP)


def friction_rows(model: mujoco.MjModel) -> int:
    from mujoco import mjx

    handle = getattr(mjx.put_model(model)._impl, "dof_hasfrictionloss", None)
    if handle is None:
        return int((model.dof_frictionloss > 0).sum())
    return int(np.asarray(handle).sum())
