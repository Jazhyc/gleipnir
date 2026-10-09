"""Fixed-axis transfer must not silently change the original ablation."""

from copy import deepcopy

import numpy as np
import pytest

from experiments.activation_filter_projection.run import validate_projection


def test_fixed_projection_rejects_layer_center_beta_and_axis_drift() -> None:
    unit = np.zeros(2560, np.float32)
    unit[0] = 1
    base = {
        "directional_edits": [
            {
                "direction": unit.tolist(),
                "layer_indices": list(range(32)),
                "beta": 1,
                "decision_centers": [0.0] * 32,
                "span_centers": [0.0] * 32,
            }
        ]
    }
    validate_projection(base, unit)
    for key, value in [
        ("layer_indices", [20]),
        ("beta", 0.25),
        ("decision_centers", [1.0] * 32),
        ("span_centers", [1.0] * 32),
        ("direction", (-unit).tolist()),
    ]:
        changed = deepcopy(base)
        changed["directional_edits"][0][key] = value
        with pytest.raises(ValueError, match="drift"):
            validate_projection(changed, unit)
