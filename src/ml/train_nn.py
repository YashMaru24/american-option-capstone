# Additional dependencies needed (merge into requirements.txt manually):
#   torch
#   pyyaml
#
# Note: `steps` is held fixed at 100 across the canonical grid and is
# therefore not used as an input feature (it carries no signal).

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn, optim

from src.ml.models import FEATURES, NormStats, build_model, save_model

SEED = 42


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split(df: pd.DataFrame, train_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    n_train = int(len(df) * train_frac)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NN pricer on binomial_price targets.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = int(config.get("seed", SEED))
    _set_seed(seed)

    features: list[str] = list(config.get("input_features", FEATURES))
    target: str = config["target"]

    df = pd.read_csv(args.data)
    train_df, test_df = _split(df, float(config["train_test_split"]), seed)

    X_train = train_df[features].to_numpy(dtype=float)
    y_train = train_df[target].to_numpy(dtype=float)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    target_mean = float(y_train.mean())
    target_std = float(y_train.std()) if y_train.std() > 0 else 1.0

    train_ranges = {f: [float(train_df[f].min()), float(train_df[f].max())] for f in features}

    X_train_norm = (X_train - mean) / std
    y_train_norm = (y_train - target_mean) / target_std

    model = build_model(config)
    optimizer = optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    loss_fn = nn.MSELoss()

    X_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_norm, dtype=torch.float32)

    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    n = X_tensor.shape[0]

    g = torch.Generator().manual_seed(seed)

    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            xb, yb = X_tensor[batch_idx], y_tensor[batch_idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)

        print(f"epoch {epoch + 1}/{epochs} loss={epoch_loss / n:.6f}")

    norm_stats = NormStats(
        mean=mean.tolist(),
        std=std.tolist(),
        target_mean=target_mean,
        target_std=target_std,
        train_ranges=train_ranges,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_model(model, args.out, norm_stats, config)

    test_csv_path = os.path.join(out_dir, "test_contracts.csv") if out_dir else "test_contracts.csv"
    test_df.to_csv(test_csv_path, index=False)


if __name__ == "__main__":
    main()
