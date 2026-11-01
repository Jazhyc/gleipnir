from experiments.adversarial_phase1_v03_replay.run import endpoint


def test_original_misses_and_new_flags_are_separate():
    r = endpoint([0.1, 0.8, 0.8], [0.3, 0.1, 0.9], 0.2)
    assert r["baseline_detected"] == 2
    assert r["baseline_missed"] == 1
    assert r["evasion"] == 1
    assert r["evasion_rate"] == 0.5
    assert r["new_detections"] == 1


def test_ties_are_not_flags_and_empty_denominator_is_null():
    r = endpoint([0.2], [0.2], 0.2)
    assert r["evasion"] == 0 and r["evasion_rate"] is None
