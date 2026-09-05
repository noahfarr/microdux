from dataclasses import dataclass

import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct

NCONMAX = 64
NJMAX = 1500

SOFTEN_SOLREF = (0.04, 1.0)
SOFTEN_SOLIMP = (0.85, 0.95, 0.001, 0.5, 2.0)


@dataclass
class Config:
    seed: int = 0
    rows: int = 8
    radius: float = 1.2
    resolution: float = 0.03
    platform_radius: float = 0.25
    ramp: float = 0.15
    max_height: float = 0.015
    base_thickness: float = 0.05
    promote_fraction: float = 0.5
    demote_fraction: float = 0.5


@dataclass
class Terrain:
    config: Config
    grid: int
    pattern: np.ndarray
    variants: np.ndarray
    amplitudes: np.ndarray
    promote_radius: float
    bounds_radius: float


def blur(field: np.ndarray) -> np.ndarray:
    padded = np.pad(field, 1, mode="edge")
    return (
        padded[:-2, 1:-1] + padded[2:, 1:-1]
        + padded[1:-1, :-2] + padded[1:-1, 2:]
        + 4.0 * padded[1:-1, 1:-1]
    ) / 8.0


def generate(config: Config = Config()) -> Terrain:
    grid = int(round(2 * config.radius / config.resolution)) + 1
    rng = np.random.default_rng(config.seed)
    noise = rng.uniform(0.0, 1.0, size=(grid, grid))
    for _ in range(3):
        noise = blur(noise)
    noise = noise - noise.min()
    peak = noise.max()
    noise = noise / peak if peak > 0 else noise

    axis = np.linspace(-config.radius, config.radius, grid)
    radial = np.sqrt(axis[None, :] ** 2 + axis[:, None] ** 2)
    mask = np.clip((radial - config.platform_radius) / max(config.ramp, 1e-9), 0.0, 1.0)
    pattern = (noise * mask).astype(np.float64)

    rows = max(config.rows, 1)
    amplitudes = config.max_height * np.arange(rows) / max(rows - 1, 1)
    scales = amplitudes / config.max_height if config.max_height > 0 else np.zeros(rows)
    variants = np.stack([(pattern * scale).reshape(-1) for scale in scales])

    return Terrain(
        config=config,
        grid=grid,
        pattern=pattern,
        variants=variants,
        amplitudes=amplitudes,
        promote_radius=config.radius * config.promote_fraction,
        bounds_radius=config.radius,
    )


def attach(spec: mujoco.MjSpec, terrain: Terrain) -> None:
    for geom in spec.geoms:
        if geom.name == "floor":
            geom.contype = 0
            geom.conaffinity = 0

    hfield = spec.add_hfield(name="terrain_hfield")
    hfield.size = [
        terrain.config.radius, terrain.config.radius,
        terrain.config.max_height, terrain.config.base_thickness,
    ]
    hfield.nrow = terrain.grid
    hfield.ncol = terrain.grid
    hfield.userdata = terrain.variants[0].copy()

    body = spec.worldbody.add_body(name="terrain")
    geom = body.add_geom(
        name="terrain_geom", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain_hfield"
    )
    geom.solref = list(SOFTEN_SOLREF)
    geom.solimp = list(SOFTEN_SOLIMP)


def sample(pattern: jnp.ndarray, radius: float, xy: jnp.ndarray) -> jnp.ndarray:
    nrow, ncol = pattern.shape
    fx = jnp.clip((xy[..., 0] + radius) / (2 * radius) * (ncol - 1), 0.0, ncol - 1)
    fy = jnp.clip((xy[..., 1] + radius) / (2 * radius) * (nrow - 1), 0.0, nrow - 1)

    x0 = jnp.floor(fx).astype(jnp.int32)
    y0 = jnp.floor(fy).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, ncol - 1)
    y1 = jnp.minimum(y0 + 1, nrow - 1)
    tx = fx - x0
    ty = fy - y0

    h00 = pattern[y0, x0]
    h10 = pattern[y0, x1]
    h01 = pattern[y1, x0]
    h11 = pattern[y1, x1]
    h0 = h00 * (1.0 - tx) + h10 * tx
    h1 = h01 * (1.0 - tx) + h11 * tx
    return h0 * (1.0 - ty) + h1 * ty


def height(pattern: jnp.ndarray, radius: float, amplitude: jnp.ndarray, xy: jnp.ndarray) -> jnp.ndarray:
    return sample(pattern, radius, xy) * amplitude


@struct.dataclass
class Progress:
    experience: jnp.ndarray
    level: jnp.ndarray


def advance(level, promoted, distance, promote_radius, done, expected_distance, rows):
    move_up = (~promoted) & (distance > promote_radius)
    promoted = promoted | move_up
    move_down = done & (~promoted) & (distance < expected_distance)
    level = jnp.clip(
        level + move_up.astype(jnp.int32) - move_down.astype(jnp.int32), 0, rows - 1
    )
    return level, promoted, move_up, move_down
