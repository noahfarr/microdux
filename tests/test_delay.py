import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microdux import delay


def test_constant_lag_returns_the_value_from_n_steps_ago():
    line = delay.Line(min_lag=2, max_lag=2, update_period=0, width=1)
    buffer = delay.rest(line, jnp.zeros(1))

    seen = []
    for step in range(6):
        buffer, out = delay.push(
            buffer, line, jnp.asarray([float(step)]), jax.random.key(step), step
        )
        seen.append(float(out[0]))

    assert seen == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]


def test_zero_lag_is_a_passthrough():
    line = delay.Line(min_lag=0, max_lag=0, update_period=0, width=1)
    buffer = delay.rest(line, jnp.zeros(1))
    for step in range(4):
        buffer, out = delay.push(
            buffer, line, jnp.asarray([float(step)]), jax.random.key(step), step
        )
        assert float(out[0]) == float(step)


def test_sampled_lag_stays_inside_the_configured_window():
    line = delay.ACTION
    seen = set()
    for seed in range(64):
        buffer = delay.start(line, jnp.zeros(line.width), jax.random.key(seed))
        seen.add(int(buffer.lag))
    assert seen
    assert min(seen) >= line.min_lag and max(seen) <= line.max_lag


def test_per_env_phase_staggers_resampling():
    line = delay.IMU
    phases = {
        int(delay.start(line, jnp.zeros(3), jax.random.key(seed)).phase)
        for seed in range(128)
    }
    assert len(phases) > 1
    assert max(phases) < line.update_period


def test_history_is_seeded_with_the_initial_value():
    line = delay.Line(min_lag=3, max_lag=3, update_period=0, width=2)
    buffer = delay.rest(line, jnp.asarray([1.0, -1.0]))
    np.testing.assert_allclose(buffer.history, np.tile([1.0, -1.0], (4, 1)))

    buffer, out = delay.push(
        buffer, line, jnp.asarray([9.0, 9.0]), jax.random.key(0), 0
    )
    np.testing.assert_allclose(out, [1.0, -1.0])


def test_jitter_stays_within_the_amplitude():
    value = jnp.zeros(64)
    noisy = delay.jitter(jax.random.key(0), value, 0.03)
    assert np.abs(np.asarray(noisy)).max() <= 0.03
    assert np.abs(np.asarray(noisy)).max() > 0.0


def test_upstream_delay_windows():
    assert (delay.IMU.min_lag, delay.IMU.max_lag, delay.IMU.update_period) == (0, 1, 64)
    assert (delay.GRAVITY.min_lag, delay.GRAVITY.max_lag) == (0, 1)
    assert (delay.JOINT_VEL.min_lag, delay.JOINT_VEL.max_lag) == (1, 1)
    assert (delay.ACTION.min_lag, delay.ACTION.max_lag) == (3, 6)
