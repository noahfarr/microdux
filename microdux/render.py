import os

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import mediapy
import mujoco
import numpy as np


def rollout(env, policy=None, steps=250, seed=0):
    policy = policy or (lambda obs, key: jnp.zeros(env.action_size))
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    key = jax.random.key(seed)
    state = reset(key)
    frames, rewards, dones = [], [], []

    for _ in range(steps):
        key, action_key = jax.random.split(key)
        action = policy(state.obs, action_key)
        state = step(state, action)
        frames.append(np.asarray(state.data.qpos))
        rewards.append(float(state.reward))
        dones.append(float(state.done))
        if dones[-1]:
            break

    return np.stack(frames), np.asarray(rewards), np.asarray(dones)


def film(env, qpos, path, width=640, height=480, camera=-1, fps=50, track=True):
    mj = env.mj_model
    data = mujoco.MjData(mj)
    renderer = mujoco.Renderer(mj, height=height, width=width)

    view = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(view)
    view.distance = 0.65
    view.elevation = -12.0
    view.azimuth = 135.0

    images = []
    for pose in qpos:
        data.qpos[:] = pose
        data.qvel[:] = 0.0
        mujoco.mj_forward(mj, data)
        if track:
            view.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, camera=view if camera < 0 else camera)
        images.append(renderer.render())

    renderer.close()
    mediapy.write_video(path, images, fps=fps)
    return path
