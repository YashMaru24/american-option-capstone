from __future__ import annotations

from src.ml.evaluate_nn import (
    count_intrinsic_violations,
    flag_extrapolation,
    intrinsic_put_value,
    put_spot_monotonicity_check,
)


def test_intrinsic_put_value_itm():
    assert intrinsic_put_value(80.0, 100.0) == 20.0


def test_intrinsic_put_value_otm():
    assert intrinsic_put_value(120.0, 100.0) == 0.0


def test_intrinsic_put_value_atm():
    assert intrinsic_put_value(100.0, 100.0) == 0.0


def test_count_intrinsic_violations_detects_violation():
    rows = [
        {"S0": 80.0, "K": 100.0, "nn_price": 15.0},  # intrinsic=20, violates
        {"S0": 80.0, "K": 100.0, "nn_price": 25.0},  # fine
    ]
    violations = count_intrinsic_violations(rows)
    assert len(violations) == 1
    assert violations[0]["nn_price"] == 15.0


def test_count_intrinsic_violations_none_when_all_valid():
    rows = [
        {"S0": 80.0, "K": 100.0, "nn_price": 20.0},
        {"S0": 120.0, "K": 100.0, "nn_price": 0.0},
    ]
    assert count_intrinsic_violations(rows) == []


def test_put_spot_monotonicity_check_no_breach_for_decreasing_fn():
    def predict_fn(S, K, T, r, sigma):
        return max(K - S, 0.0)

    breaches = put_spot_monotonicity_check(predict_fn)
    assert breaches == []


def test_put_spot_monotonicity_check_detects_breach():
    def predict_fn(S, K, T, r, sigma):
        # Artificially increasing with S -> should be flagged.
        return S

    breaches = put_spot_monotonicity_check(predict_fn)
    assert len(breaches) > 0


def test_flag_extrapolation_within_range():
    row = {"S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.25}
    train_ranges = {"S0": [70.0, 130.0], "K": [100.0, 100.0], "T": [0.25, 2.0],
                     "r": [0.02, 0.05], "sigma": [0.15, 0.40]}
    assert flag_extrapolation(row, train_ranges) is False


def test_flag_extrapolation_outside_range():
    row = {"S0": 200.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.25}
    train_ranges = {"S0": [70.0, 130.0], "K": [100.0, 100.0], "T": [0.25, 2.0],
                     "r": [0.02, 0.05], "sigma": [0.15, 0.40]}
    assert flag_extrapolation(row, train_ranges) is True
