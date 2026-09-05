import dataclasses
import inspect
from typing import Any, Dict, Optional, Union

from ml_collections import config_dict

from . import terrain as rough
from .ballkick import BallKick
from .env import SitStand, StandUp, Velocity
from .groundpick import GroundPick
from .rollercrouch import RollerCrouch
from .rollers import Rollers
from .rollerslope import RollerSlope
from .rollerstandup import RollerStandUp
from .roulade import Roulade
from .spin import Spin
from .swizzle import Swizzle
from .velstand import VelStand

TASKS = {
    "MicroduckVelocity": (Velocity, {}),
    "MicroduckRough": (Velocity, {"terrain": rough.Config()}),
    "MicroduckVelStand": (VelStand, {}),
    "MicroduckStandUp": (StandUp, {}),
    "MicroduckSitStand": (SitStand, {}),
    "MicroduckSpin": (Spin, {}),
    "MicroduckSwizzle": (Swizzle, {}),
    "MicroduckRollers": (Rollers, {}),
    "MicroduckRollerCrouch": (RollerCrouch, {}),
    "MicroduckRollerStandUp": (RollerStandUp, {}),
    "MicroduckRollerSlope": (RollerSlope, {}),
    "MicroduckRoulade": (Roulade, {}),
    "MicroduckBallKick": (BallKick, {}),
    "MicroduckGroundPick": (GroundPick, {}),
}

PLAIN = (int, float, bool, str)


def _spread(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        nested = config_dict.ConfigDict()
        for field in dataclasses.fields(value):
            entry = getattr(value, field.name)
            if isinstance(entry, PLAIN):
                nested[field.name] = entry
        return nested if len(nested) else None
    return None


def default_config(name: str) -> config_dict.ConfigDict:
    cls, fixed = TASKS[name]
    config = config_dict.ConfigDict()
    for parameter in inspect.signature(cls.__init__).parameters.values():
        if parameter.name in ("self", "envs", "impl", "nconmax", "njmax"):
            continue
        value = fixed.get(parameter.name, parameter.default)
        if value is inspect.Parameter.empty:
            continue
        if isinstance(value, PLAIN):
            config[parameter.name] = value
            continue
        if value is None and parameter.name in ("weights", "tuning", "ranges"):
            value = _instance(cls, parameter.name)
        nested = _spread(value)
        if nested is not None:
            config[parameter.name] = nested
    return config


def _instance(cls, name):
    hint = {"weights": "Weights", "tuning": "Tuning", "ranges": "Ranges"}[name]
    module = inspect.getmodule(cls)
    for suffix in (cls.__name__ + hint, hint):
        made = getattr(module, suffix, None)
        if made is not None:
            return made()
    from . import commands, rewards

    for source in (rewards, commands):
        made = getattr(source, cls.__name__ + hint, None) or getattr(source, hint, None)
        if made is not None:
            return made()
    return None


def _apply(cls, fixed, config):
    kwargs = dict(fixed)
    for key, value in config.items():
        if isinstance(value, config_dict.ConfigDict):
            seed = _instance(cls, key)
            if seed is not None:
                kwargs[key] = dataclasses.replace(seed, **dict(value))
        else:
            kwargs[key] = value
    return kwargs


def load(
    name: str,
    config: Optional[config_dict.ConfigDict] = None,
    config_overrides: Optional[Dict[str, Union[str, int, Any]]] = None,
    **extra,
):
    cls, fixed = TASKS[name]
    config = config or default_config(name)
    if config_overrides:
        config = config_dict.ConfigDict(config)
        config.update_from_flattened_dict(config_overrides)
    return cls(**_apply(cls, fixed, config), **extra)


def register() -> None:
    from mujoco_playground._src import locomotion

    for name in TASKS:
        locomotion.register_environment(
            name,
            _factory(name),
            (lambda held=name: default_config(held)),
        )


def _factory(name):
    def make(config=None, config_overrides=None):
        return load(name, config, config_overrides)

    return make
