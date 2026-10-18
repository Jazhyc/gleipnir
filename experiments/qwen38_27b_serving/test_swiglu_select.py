from experiments.qwen38_27b_serving.fp4_swiglu_native_canary import ROWS, select


def results(tile, ratios, passed=True):
    return [
        {"tile": tile, "m": m, "passed": passed, "full_mlp_speed_ratio": r}
        for m, r in zip(ROWS, ratios, strict=True)
    ]


def test_band_is_the_longest_faster_run():
    fast = results("256x128", [0.99, 1.08, 1.12, 1.08, 1.04, 1.13, 1.10, 0.99, 0.96])
    slow = results("256x256", [0.96, 0.98, 0.99, 0.86, 0.86, 0.94, 1.01, 0.92, 0.90])
    choice = select(fast + slow)
    assert choice["tile"] == "256x128"
    assert (choice["minimum_rows"], choice["maximum_rows"]) == (17, 4096)


def test_failed_tiles_and_no_gain_select_nothing():
    failed = results("256x128", [1.5] * len(ROWS), passed=False)
    assert select(failed)["tile"] is None
    flat = select(results("256x128", [1.0] * len(ROWS)))
    assert flat["tile"] == "256x128" and flat["minimum_rows"] is None
