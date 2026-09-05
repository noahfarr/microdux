from flax import struct


def inner(handle):
    return getattr(handle, "_impl", handle)


@struct.dataclass
class Contacts:
    geom1: object
    geom2: object
    dist: object
    efc_address: object
    friction: object
    frame: object
    dim: object


def contacts(data) -> Contacts:
    core = inner(data)
    held = getattr(core, "contact", None)
    if held is not None:
        return Contacts(
            geom1=held.geom1,
            geom2=held.geom2,
            dist=held.dist,
            efc_address=held.efc_address,
            friction=held.friction,
            frame=held.frame,
            dim=held.dim,
        )

    geom = core.contact__geom
    return Contacts(
        geom1=geom[..., 0],
        geom2=geom[..., 1],
        dist=core.contact__dist,
        efc_address=core.contact__efc_address,
        friction=core.contact__friction,
        frame=core.contact__frame,
        dim=core.contact__dim,
    )


def efc_force(data):
    core = inner(data)
    found = getattr(core, "efc_force", None)
    return core.efc__force if found is None else found


def efc_type(data):
    core = inner(data)
    found = getattr(core, "efc_type", None)
    return core.efc__type if found is None else found
