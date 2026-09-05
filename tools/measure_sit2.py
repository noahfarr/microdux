import numpy as np
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import sys
sys.path.insert(0, "/home/farr/microdux")
from microdux import model as mmodel, constants, actuator, plant

mj_model, layout = mmodel.build("walk", timestep=0.005, iterations=10, ls_iterations=20)
mjx_model = mmodel.to_mjx(mj_model, "jax")
wiring = plant.wire(mj_model, layout)
bam = actuator.load(kp=200.0)

names = layout.actuators
sit = dict(constants.HOME)
sit.update({
    r".*left_hip_roll.*": 0.0,
    r".*right_hip_roll.*": 0.0,
    r".*left_hip_pitch.*": -0.4079,
    r".*right_hip_pitch.*": 0.4079,
    r".*left_knee.*": 1.35,
    r".*right_knee.*": -1.35,
    r".*left_ankle.*": 0.0,
    r".*right_ankle.*": 0.0,
})
targets = jnp.asarray(constants.resolve(sit, names))

qpos = np.zeros(mj_model.nq)
qpos[3] = 1.0
qpos[2] = 0.09
qpos[layout.actuated_qpos_adr] = np.asarray(targets)

data = mjx.make_data(mj_model)
data = data.replace(qpos=jnp.asarray(qpos))
data = mjx.forward(mjx_model, data)

servos = plant.rest(mj_model.nu)

for i in range(400):
    data, servos = plant.substep(mjx_model, data, servos, bam, targets, wiring)
    if i % 50 == 0:
        z = float(data.qpos[2])
        quat = np.asarray(data.qpos[3:7])
        angle = 2 * np.degrees(np.arccos(np.clip(abs(quat[0]), -1, 1)))
        print(i, "z=", z, "tilt_deg=", angle)

z = float(data.qpos[2])
quat = np.asarray(data.qpos[3:7])
angle = 2 * np.degrees(np.arccos(np.clip(abs(quat[0]), -1, 1)))
print("final z=", z, "tilt_deg=", angle, "qvel norm", float(jnp.linalg.norm(data.qvel)))
