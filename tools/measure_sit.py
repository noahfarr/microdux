import numpy as np
import mujoco
import sys
sys.path.insert(0, "/home/farr/microdux")
from microdux import model as mmodel
from microdux import constants

mj_model, layout = mmodel.build("walk", timestep=0.002, iterations=10, ls_iterations=20)
data = mujoco.MjData(mj_model)

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
targets = np.asarray(constants.resolve(sit, names))

qpos_adr = layout.actuated_qpos_adr
qvel_adr = layout.actuated_qvel_adr

data.qpos[:] = 0
data.qpos[2] = 0.14
data.qpos[3] = 1.0
data.qpos[qpos_adr] = targets
mujoco.mj_forward(mj_model, data)

zs = []
tilts = []
for i in range(4000):
    data.qpos[qpos_adr] = targets
    data.qvel[qvel_adr] = 0.0
    mujoco.mj_step(mj_model, data)
    if i > 3000:
        zs.append(data.qpos[2])
        tilts.append(data.qpos[3:7].copy())

print("z mean last 1000:", np.mean(zs), "std:", np.std(zs))
print("final quat:", data.qpos[3:7])
print("final z:", data.qpos[2])
