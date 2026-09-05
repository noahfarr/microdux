import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

TOOLS = Path(__file__).parent.parent / "tools"
CHECKPOINT = Path(__file__).parent.parent / "runs" / "faithful" / "000307593216"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def export_onnx():
    return load_tool("export_onnx")


@pytest.fixture(scope="module")
def checkpoint():
    if not CHECKPOINT.exists():
        pytest.skip("no local ppo checkpoint fixture present")
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    return ppo_checkpoint.load(str(CHECKPOINT))


def forward(layer_specs, obs):
    x = obs
    for kernel, bias, activation in layer_specs:
        x = x @ jnp.asarray(kernel) + jnp.asarray(bias)
        if activation == "Elu":
            x = jax.nn.elu(x)
    return x


def test_unpack_reads_the_checkpoint_shapes(export_onnx, checkpoint):
    mean, std, trunk, head = export_onnx.unpack(checkpoint)
    assert mean.shape == (61,)
    assert std.shape == (61,)
    assert trunk["hidden_0"]["kernel"].shape == (61, 512)
    assert trunk["hidden_1"]["kernel"].shape == (512, 256)
    assert trunk["hidden_2"]["kernel"].shape == (256, 128)
    assert head["kernel"].shape == (128, 14)


def test_folded_layers_reproduce_the_brax_policy_mean(export_onnx, checkpoint):
    from microdux.train import networks, policy_from
    from microdux import Velocity

    env = Velocity()
    act = policy_from(env, checkpoint, deterministic=True)

    key = jax.random.key(0)
    obs = jax.random.normal(key, (5, 61)) * 0.1

    layer_specs = export_onnx.layers(checkpoint, obs_width=61)
    ours = forward(layer_specs, obs)

    reference = jax.vmap(lambda o: act({"state": o}, jax.random.key(0)))(obs)

    np.testing.assert_allclose(np.asarray(ours), np.asarray(reference), rtol=1e-5, atol=1e-5)


def test_folded_layers_react_to_the_activation_choice(export_onnx, checkpoint):
    layer_specs = export_onnx.layers(checkpoint, obs_width=61)
    obs = jnp.zeros((1, 61))

    correct = forward(layer_specs, obs)
    relu_specs = [
        (k, b, "Relu" if a == "Elu" else a) for k, b, a in layer_specs
    ]

    def forward_relu(specs, obs):
        x = obs
        for kernel, bias, activation in specs:
            x = x @ jnp.asarray(kernel) + jnp.asarray(bias)
            if activation == "Relu":
                x = jax.nn.relu(x)
        return x

    wrong = forward_relu(relu_specs, obs)
    assert not np.allclose(np.asarray(correct), np.asarray(wrong))


def test_metadata_matches_the_layout(export_onnx):
    from microdux import model

    fields = export_onnx.metadata("walk", action_scale=1.0, ctrl_dt=0.02, run_path="x", bounded=False)
    _, layout = model.build("walk")

    assert fields["joint_names"] == list(layout.actuators)
    assert len(fields["joint_names"]) == 14
    assert len(fields["default_joint_pos"]) == 14
    np.testing.assert_allclose(fields["default_joint_pos"], layout.home, atol=1e-6)
    assert fields["action_scale"] == 1.0


@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None or importlib.util.find_spec("onnxruntime") is None,
    reason="onnx / onnxruntime not installed in this environment",
)
def test_exported_onnx_graph_reproduces_the_policy(export_onnx, checkpoint, tmp_path):
    import onnxruntime as ort

    from microdux.train import policy_from
    from microdux import Velocity

    out = str(tmp_path / "policy.onnx")
    export_onnx.export(checkpoint, out, variant="walk", action_scale=1.0, ctrl_dt=0.02)

    env = Velocity()
    act = policy_from(env, checkpoint, deterministic=True)

    key = jax.random.key(0)
    obs = np.asarray(jax.random.normal(key, (3, 61)) * 0.1, dtype=np.float32)

    session = ort.InferenceSession(out)
    (got,) = session.run(["action"], {"observation": obs})

    reference = jax.vmap(lambda o: act({"state": o}, jax.random.key(0)))(jnp.asarray(obs))
    np.testing.assert_allclose(got, np.asarray(reference), rtol=1e-5, atol=1e-5)
