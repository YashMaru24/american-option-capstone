from __future__ import annotations

import argparse
import os
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.ml.models import NNPricer


def intrinsic_put_value(S: float, K: float) -> float:
    return max(K - S, 0.0)


def count_intrinsic_violations(rows: list[dict]) -> list[dict]:
    violations = []
    for row in rows:
        intrinsic = intrinsic_put_value(row["S0"], row["K"])
        if row["nn_price"] + 1e-8 < intrinsic:
            violations.append(row)
    return violations


def put_spot_monotonicity_check(
    predict_fn: Callable[[float, float, float, float, float], float],
    K: float = 100,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.25,
) -> list[tuple]:
    spots = np.linspace(60, 140, 41)
    prices = [predict_fn(S, K, T, r, sigma) for S in spots]

    breaches = []
    for i in range(1, len(spots)):
        delta = prices[i] - prices[i - 1]
        if delta > 1e-6:
            breaches.append((float(spots[i - 1]), float(spots[i]), float(delta)))
    return breaches


def flag_extrapolation(row: dict, train_ranges: dict) -> bool:
    for feature, (lo, hi) in train_ranges.items():
        if feature in row and (row[feature] < lo or row[feature] > hi):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NN pricer against binomial reference.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    pricer = NNPricer(args.model)
    train_ranges = pricer.norm_stats.train_ranges

    df = pd.read_csv(args.data)
    nn_prices = pricer.predict_batch(df)

    result = df.copy()
    result["nn_price"] = nn_prices
    result["nn_error"] = result["nn_price"] - result["binomial_price"]

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.to_csv(args.out, index=False)

    rows = result.to_dict("records")
    violations = count_intrinsic_violations(rows)
    extrapolated = [r for r in rows if flag_extrapolation(r, train_ranges)]

    def predict_fn(S: float, K: float, T: float, r: float, sigma: float) -> float:
        X = np.array([[S, K, T, r, sigma]], dtype=float)
        return float(pricer.predict_array(X)[0])

    breaches = put_spot_monotonicity_check(predict_fn)

    print(f"Rows evaluated: {len(rows)}")
    print(f"Intrinsic-value violations: {len(violations)}")
    print(f"Extrapolated rows (outside training range): {len(extrapolated)}")
    print(f"Monotonicity breaches: {len(breaches)}")


if __name__ == "__main__":
    main()
