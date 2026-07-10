from __future__ import annotations

from typing import Any, Optional

from src.contract import OptionContract


class PricingResult:
    def __init__(self, price: float, metadata: Optional[dict[str, Any]] = None) -> None:
        self.price = price
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"PricingResult(price={self.price!r}, metadata={self.metadata!r})"


class Pricer:
    name = "base"

    def price(self, contract: OptionContract) -> PricingResult:
        raise NotImplementedError
