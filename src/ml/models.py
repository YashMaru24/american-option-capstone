from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.contract import OptionContract

FEATURES = ["S0", "K", "T", "r", "sigma"]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layer_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_model(config: dict) -> MLP:
    torch.manual_seed(int(config.get("seed", 42)))
    return MLP(input_dim=len(config.get("input_features", FEATURES)),
               hidden_layer_sizes=list(config["hidden_layer_sizes"]))


@dataclass
class NormStats:
    mean: list[float]
    std: list[float]
    target_mean: float
    target_std: float
    train_ranges: dict[str, list[float]]


def save_model(model: MLP, path: str, norm_stats: NormStats, config: dict) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_layer_sizes": list(config["hidden_layer_sizes"]),
            "input_features": list(config.get("input_features", FEATURES)),
            "norm_stats": asdict(norm_stats),
            "config": config,
        },
        path,
    )


def load_model(path: str) -> tuple[MLP, NormStats, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = MLP(
        input_dim=len(checkpoint["input_features"]),
        hidden_layer_sizes=checkpoint["hidden_layer_sizes"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    norm_stats = NormStats(**checkpoint["norm_stats"])
    return model, norm_stats, checkpoint["config"]


class NNPricer:
    """Uniform inference wrapper for the trained NN pricer."""

    def __init__(self, path: str) -> None:
        self.model, self.norm_stats, self.config = load_model(path)
        self.features: list[str] = self.config.get("input_features", FEATURES)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.norm_stats.mean, dtype=float)
        std = np.asarray(self.norm_stats.std, dtype=float)
        return (X - mean) / std

    def _denormalize_target(self, y: np.ndarray) -> np.ndarray:
        return y * self.norm_stats.target_std + self.norm_stats.target_mean

    def predict(self, contract: OptionContract) -> float:
        row = {"S0": contract.S0, "K": contract.K, "T": contract.T,
               "r": contract.r, "sigma": contract.sigma}
        X = np.array([[row[f] for f in self.features]], dtype=float)
        return float(self.predict_array(X)[0])

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        X_norm = self._normalize(X)
        with torch.no_grad():
            y_norm = self.model(torch.tensor(X_norm, dtype=torch.float32)).numpy()
        return self._denormalize_target(y_norm)

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(dtype=float)
        return self.predict_array(X)
