import mujoco
import numpy as np
import pytest

from microdux import actuator, constants, model


@pytest.fixture(scope="module")
def built():
    return model.build("walk")


def test_friction_rows_survive_the_bam_conversion(built):
    mj, layout = built
    assert model.friction_rows(mj) == len(layout.actuators) == 14

    zeroed = mujoco.MjModel.from_xml_path(str(constants.XMLS / "scene_walk.xml"))
    zeroed.dof_frictionloss[:] = 0.0
    assert model.friction_rows(zeroed) == 0


def test_actuators_are_torque_sources(built):
    mj, layout = built
    bam = actuator.load(kp=200.0)
    assert mj.nu == 14
    assert set(mj.actuator_trntype.tolist()) == {int(mujoco.mjtTrn.mjTRN_JOINT)}
    np.testing.assert_allclose(mj.actuator_gear[:, 0], 1.0)

    limit = 8.2 * bam.kt / bam.resistance
    np.testing.assert_allclose(mj.actuator_forcerange[:, 1], limit, rtol=1e-6)
    np.testing.assert_allclose(mj.actuator_forcerange[:, 0], -limit, rtol=1e-6)


def test_joint_armature_and_damping_come_from_bam(built):
    mj, layout = built
    bam = actuator.load(kp=200.0)
    np.testing.assert_allclose(mj.dof_armature[layout.actuated_qvel_adr], bam.armature, rtol=1e-9)
    np.testing.assert_allclose(mj.dof_damping[layout.actuated_qvel_adr], 0.0, atol=0)


def test_home_pose_matches_the_stand_keyframe(built):
    mj, layout = built
    stand = mj.key_qpos[layout.keyframes["STAND"]][layout.actuated_qpos_adr]
    np.testing.assert_allclose(layout.home, stand, atol=1e-4)


def test_collision_set_matches_upstream(built):
    mj, layout = built
    colliding = [
        i for i in range(mj.ngeom) if mj.geom_contype[i] or mj.geom_conaffinity[i]
    ]
    assert len(colliding) == 6
    assert set(layout.foot_geoms).issubset(colliding)
    assert layout.floor_geom in colliding
    assert (mj.geom_contype[layout.foot_geoms] != 0).all()


def test_hulls_replace_the_undecimated_collision_meshes(built):
    mj, layout = built
    for geom in layout.foot_geoms:
        assert mj.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
        assert mj.mesh_vertnum[mj.geom_dataid[geom]] < 64

    raw, _ = model.build("walk", hulls=False)
    for geom in layout.foot_geoms:
        assert raw.mesh_vertnum[raw.geom_dataid[geom]] > 5000


def test_only_colliders_are_decimated(built):
    mj, _ = built
    raw, _ = model.build("walk", hulls=False)
    changed = [
        i for i in range(mj.ngeom)
        if mj.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        and mj.mesh_vertnum[mj.geom_dataid[i]] != raw.mesh_vertnum[raw.geom_dataid[i]]
    ]
    assert changed, "no geom was decimated"
    for geom in changed:
        assert mj.geom_contype[geom] or mj.geom_conaffinity[geom], (
            f"geom {geom} is non-colliding yet was decimated, which degrades rendering"
        )


def test_every_variant_builds():
    for variant in model.SCENES:
        mj, layout = model.build(variant)
        assert mj.nu >= 14
        assert len(layout.home) == mj.nu
        assert model.friction_rows(mj) >= 14


def test_walk_variants_have_foot_colliders():
    for variant in ("walk", "allcollisions", "walk_backlash", "backlash"):
        _, layout = model.build(variant)
        assert len(layout.foot_geoms) == 2, variant
        assert len(layout.foot_sites) == 2, variant


def test_hull_colliders_sit_where_upstream_puts_them():
    import json
    from pathlib import Path

    golden = json.loads((Path(__file__).parent / "golden_colliders.json").read_text())

    for variant, expected in golden.items():
        mj, _ = model.build(variant)
        data = mujoco.MjData(mj)
        key = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        mujoco.mj_resetDataKeyframe(mj, data, key)
        mujoco.mj_forward(mj, data)

        for name, frame in expected["geoms"].items():
            gid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, name)
            np.testing.assert_allclose(
                data.geom_xpos[gid], frame["pos"], atol=2e-5,
                err_msg=f"{variant}/{name} collider moved",
            )
            np.testing.assert_allclose(data.geom_xmat[gid], frame["mat"], atol=2e-5)

        assert int(data.ncon) == expected["contacts"], variant


def test_pose_resolution_refuses_an_unmatched_joint():
    with pytest.raises(KeyError):
        constants.resolve({r"nothing": 0.0}, ["left_knee"])


CRITICAL = (
    "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
    "body_parentid", "body_jntnum", "body_dofnum",
    "jnt_type", "jnt_axis", "jnt_pos", "jnt_range", "jnt_limited", "jnt_stiffness",
    "jnt_bodyid", "jnt_qposadr", "jnt_dofadr",
    "geom_friction", "geom_solref", "geom_solimp", "geom_condim", "geom_priority",
    "geom_margin", "geom_gap", "geom_bodyid", "geom_type",
    "site_pos", "site_quat", "site_bodyid",
    "actuator_trnid", "actuator_ctrlrange",
    "qpos0", "key_qpos",
)


def test_physics_inputs_are_identical_to_upstream(built):
    mj, _ = built
    upstream = mujoco.MjModel.from_xml_path(str(constants.XMLS / "scene_walk.xml"))

    for field in CRITICAL:
        theirs = np.asarray(getattr(upstream, field))
        ours = np.asarray(getattr(mj, field))
        assert theirs.shape == ours.shape, field
        np.testing.assert_array_equal(theirs, ours, err_msg=f"{field} diverged")

    np.testing.assert_array_equal(upstream.opt.gravity, mj.opt.gravity)


def test_the_only_solver_changes_are_the_ones_mjlab_makes(built):
    mj, _ = built
    upstream = mujoco.MjModel.from_xml_path(str(constants.XMLS / "scene_walk.xml"))
    assert mj.opt.timestep == 0.005
    assert mj.opt.iterations == 10
    assert mj.opt.ls_iterations == 20
    assert upstream.opt.timestep != mj.opt.timestep
