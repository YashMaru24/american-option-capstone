from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd

from src.contract import OptionContract
from src.pricing.binomial import BinomialAmericanPutPricer


def make_contract_grid() -> pd.DataFrame:
    spots = [70, 80, 90, 100, 110, 120, 130]
    strikes = [100]
    maturities = [0.25, 0.5, 1.0, 2.0]
    rates = [0.02, 0.05]
    sigmas = [0.15, 0.25, 0.40]
    steps = 100

    rows = []
    for S0, K, T, r, sigma in itertools.product(spots, strikes, maturities, rates, sigmas):
        rows.append({"S0": S0, "K": K, "T": T, "r": r, "sigma": sigma, "steps": steps})

    return pd.DataFrame(rows, columns=["S0", "K", "T", "r", "sigma", "steps"])


def train_test_split_contracts(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split contracts into train/test sets.

    Uses sklearn.model_selection.train_test_split with a fixed random_state
    of 42 for full reproducibility across sessions.
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _compute_binomial_prices(df: pd.DataFrame) -> pd.DataFrame:
    pricer = BinomialAmericanPutPricer()
    prices = []
    for row in df.itertuples(index=False):
        contract = OptionContract(
            S0=float(row.S0),
            K=float(row.K),
            T=float(row.T),
            r=float(row.r),
            sigma=float(row.sigma),
            steps=int(row.steps),
            option_type="put",
        )
        result = pricer.price(contract)
        prices.append(result.price)

    out = df.copy()
    out["binomial_price"] = prices
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic option contract grid.")
    parser.add_argument("--out", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()

    grid = make_contract_grid()
    grid = _compute_binomial_prices(grid)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    grid.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
