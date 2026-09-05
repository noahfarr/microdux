import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from torelax import fold

from microdux import actuator, model

HIDDEN = (512, 256, 128)
OBSERVATION_NAMES = (
    "angular_velocity", "gravity", "joint_pos", "joint_vel", "last_action", "commands",
)
COMMAND_NAMES = ("twist", "head", "body")


def unpack(source):
    normaliser, policy, _ = source
    mean = normaliser.mean["state"]
    std = np.maximum(np.asarray(normaliser.std["state"]), 1e-8)
    trunk = policy["params"]["MLP_0"]
    head = policy["params"]["Dense_0"]
    return np.asarray(mean), std, trunk, head


def layers(source, obs_width: int):
    mean, std, trunk, head = unpack(source)
    kernel, bias = fold(trunk["hidden_0"]["kernel"], trunk["hidden_0"]["bias"], mean, std, obs_width)
    return [
        (np.asarray(kernel), np.asarray(bias), "Elu"),
        (np.asarray(trunk["hidden_1"]["kernel"]), np.asarray(trunk["hidden_1"]["bias"]), "Elu"),
        (np.asarray(trunk["hidden_2"]["kernel"]), np.asarray(trunk["hidden_2"]["bias"]), "Elu"),
        (np.asarray(head["kernel"]), np.asarray(head["bias"]), None),
    ]


def graph(layer_specs, obs_width: int, bounded: bool):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes, initializers = [], []
    x = "observation"
    for index, (kernel, bias, activation) in enumerate(layer_specs):
        w_name, b_name = f"kernel_{index}", f"bias_{index}"
        initializers.append(numpy_helper.from_array(kernel.astype(np.float32), w_name))
        initializers.append(numpy_helper.from_array(bias.astype(np.float32), b_name))

        y = f"gemm_{index}"
        nodes.append(helper.make_node("Gemm", [x, w_name, b_name], [y], alpha=1.0, beta=1.0))
        x = y

        if activation is not None:
            z = f"activation_{index}"
            nodes.append(helper.make_node(activation, [x], [z]))
            x = z

    if bounded:
        nodes.append(helper.make_node("Tanh", [x], ["action"]))
    else:
        nodes[-1].output[0] = "action"

    actions = layer_specs[-1][0].shape[1]
    inputs = [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", obs_width])]
    outputs = [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["batch", actions])]

    made = helper.make_graph(nodes, "microdux_policy", inputs, outputs, initializers)
    return helper.make_model(made, opset_imports=[helper.make_opsetid("", 17)])


def metadata(variant: str, action_scale: float, ctrl_dt: float, run_path: str, bounded: bool):
    _, layout = model.build(variant)
    bam = actuator.load(kp=200.0)
    joints = list(layout.actuators)
    return {
        "run_path": run_path,
        "joint_names": joints,
        "joint_stiffness": [float(bam.kp)] * len(joints),
        "joint_damping": [float(bam.friction_viscous)] * len(joints),
        "default_joint_pos": [float(x) for x in layout.home],
        "command_names": list(COMMAND_NAMES),
        "observation_names": list(OBSERVATION_NAMES),
        "action_scale": float(action_scale),
        "control_dt": float(ctrl_dt),
        "bounded": str(bounded),
    }


def attach(onnx_path: str, fields: dict) -> None:
    import onnx

    made = onnx.load(onnx_path)
    for key, value in fields.items():
        entry = made.metadata_props.add()
        entry.key = key
        entry.value = ",".join(str(v) for v in value) if isinstance(value, list) else str(value)
    onnx.save(made, onnx_path)


def export(source, out_path: str, variant: str = "walk", action_scale: float = 1.0,
           ctrl_dt: float = 0.02, run_path: str = "", bounded: bool = False,
           obs_width: int | None = None):
    import onnx

    mean, _, _, _ = unpack(source)
    width = obs_width or mean.shape[0]

    made = graph(layers(source, width), width, bounded)
    onnx.save(made, out_path)
    attach(out_path, metadata(variant, action_scale, ctrl_dt, run_path, bounded))
    return out_path


if __name__ == "__main__":
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    checkpoint, out = sys.argv[1], sys.argv[2]
    variant = sys.argv[3] if len(sys.argv) > 3 else "walk"

    export(ppo_checkpoint.load(checkpoint), out, variant=variant)
    print("wrote", out)
