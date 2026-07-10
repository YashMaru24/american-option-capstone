from dataclasses import dataclass


@dataclass(frozen=True)
class OptionContract:
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    steps: int = 100
    option_type: str = "put"

    def validate(self) -> None:
        assert self.S0 > 0
        assert self.K > 0
        assert self.T > 0
        assert self.sigma > 0
        assert self.steps >= 1
        assert self.option_type in {"put", "call"}
