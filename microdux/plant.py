import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct
from mujoco import mjx

from . import actuator as bam_actuator
from . import backend

FRICTION_DOF = 1


@struct.dataclass
class Servos:
    vin: jax.Array
    vin_drop_gain: jax.Array
    kp_scale: jax.Array
    kd_scale: jax.Array
    friction_scale: jax.Array
    previous_motor_torque: jax.Array


@struct.dataclass
class Wiring:
    qpos_adr: np.ndarray = struct.field(pytree_node=False)
    qvel_adr: np.ndarray = struct.field(pytree_node=False)
    friction_dofs: np.ndarray = struct.field(pytree_node=False)
    friction_rows: np.ndarray = struct.field(pytree_node=False)
    nv: int = struct.field(pytree_node=False)
    vin_min: float = struct.field(pytree_node=False)
    backlash_qpos_adr: np.ndarray = struct.field(pytree_node=False, default=None)
    backlash_qvel_adr: np.ndarray = struct.field(pytree_node=False, default=None)
    backlash_mask: np.ndarray = struct.field(pytree_node=False, default=None)


def impl(data):
    return getattr(data, "_impl", data)


EQUALITY_ROWS = {
    int(mujoco.mjtEq.mjEQ_CONNECT): 3,
    int(mujoco.mjtEq.mjEQ_WELD): 6,
    int(mujoco.mjtEq.mjEQ_JOINT): 1,
    int(mujoco.mjtEq.mjEQ_TENDON): 1,
}


def rows(mj_model) -> tuple[np.ndarray, np.ndarray]:
    friction_dofs = np.nonzero(np.asarray(mj_model.dof_frictionloss) > 0)[0]
    equalities = sum(
        EQUALITY_ROWS.get(int(kind), 0) for kind in np.asarray(mj_model.eq_type)
    )
    return friction_dofs, equalities + np.arange(friction_dofs.size)


def solved_rows(mjx_model) -> np.ndarray:
    data = mjx.forward(mjx_model, mjx.make_data(mjx_model))
    return np.nonzero(np.asarray(backend.efc_type(data)) == FRICTION_DOF)[0]


def wire(mj_model, layout, vin_min: float = 6.0) -> Wiring:
    friction_dofs, friction_rows = rows(mj_model)

    if friction_rows.size == 0:
        raise RuntimeError(
            "no friction constraint rows; the per-step friction writes would not "
            "reach the solver"
        )

    return Wiring(
        qpos_adr=np.asarray(layout.actuated_qpos_adr),
        qvel_adr=np.asarray(layout.actuated_qvel_adr),
        friction_dofs=friction_dofs,
        friction_rows=friction_rows,
        nv=int(mj_model.nv),
        vin_min=vin_min,
        backlash_qpos_adr=np.asarray(layout.backlash_qpos_adr),
        backlash_qvel_adr=np.asarray(layout.backlash_qvel_adr),
        backlash_mask=np.asarray(layout.backlash_mask),
    )


def readout(raw, adr, backlash_adr, backlash_mask):
    values = raw[adr]
    if backlash_adr is None:
        return values
    return values + raw[backlash_adr] * backlash_mask


def loads(data, wiring: Wiring):
    friction = jnp.zeros(wiring.nv).at[wiring.friction_dofs].set(
        backend.efc_force(data)[wiring.friction_rows]
    )
    dofs = wiring.qvel_adr
    external = -data.qfrc_bias[dofs] + data.qfrc_constraint[dofs] - friction[dofs]
    return external, data.qfrc_actuator[dofs]


def substep(mjx_model, data, servos: Servos, bam: bam_actuator.Bam,
            target, wiring: Wiring):
    external, previous_actuator = loads(data, wiring)

    position = readout(data.qpos, wiring.qpos_adr, wiring.backlash_qpos_adr, wiring.backlash_mask)
    out = bam_actuator.drive(
        bam,
        target=target,
        position=position,
        velocity=data.qvel[wiring.qvel_adr],
        external_torque=external,
        previous_actuator_torque=previous_actuator,
        previous_motor_torque=servos.previous_motor_torque,
        vin=servos.vin,
        kp_scale=servos.kp_scale,
        kd_scale=servos.kd_scale,
        friction_scale=servos.friction_scale,
        vin_drop_gain=servos.vin_drop_gain,
        vin_min=wiring.vin_min,
    )

    stepped = mjx_model.tree_replace({
        "dof_frictionloss": mjx_model.dof_frictionloss.at[wiring.qvel_adr].set(out.frictionloss),
        "dof_damping": mjx_model.dof_damping.at[wiring.qvel_adr].set(out.damping),
    })
    data = mjx.step(stepped, data.replace(ctrl=out.torque))
    return data, servos.replace(previous_motor_torque=out.motor_torque)


def advance(mjx_model, data, servos: Servos, bam: bam_actuator.Bam,
            target, wiring: Wiring, substeps: int, history=None, lag=None):
    if history is None:
        def once(carry, _):
            data, servos = carry
            return substep(mjx_model, data, servos, bam, target, wiring), None

        (data, servos), _ = jax.lax.scan(once, (data, servos), (), substeps)
        return data, servos, None

    def delayed(carry, _):
        data, servos, history = carry
        history = jnp.roll(history, 1, axis=0).at[0].set(target)
        held = history[lag]
        data, servos = substep(mjx_model, data, servos, bam, held, wiring)
        return (data, servos, history), None

    (data, servos, history), _ = jax.lax.scan(
        delayed, (data, servos, history), (), substeps
    )
    return data, servos, history


def rest(nu: int, vin: float = 7.5, vin_drop_gain: float = 0.0) -> Servos:
    return Servos(
        vin=jnp.asarray([vin]),
        vin_drop_gain=jnp.asarray([vin_drop_gain]),
        kp_scale=jnp.ones(1),
        kd_scale=jnp.ones(1),
        friction_scale=jnp.ones(1),
        previous_motor_torque=jnp.zeros(nu),
    )
