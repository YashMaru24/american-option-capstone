import pytest

from src.contract import OptionContract
from src.pricing.binomial import crr_american_put_with_boundary, crr_price
from src.pricing.black_scholes import black_scholes_price


def _contract(**overrides):
    defaults = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=100, option_type="put")
    defaults.update(overrides)
    return OptionContract(**defaults)


@pytest.mark.parametrize(
    "S0,K,T,r,sigma",
    [
        (80.0, 100.0, 1.0, 0.05, 0.25),
        (100.0, 100.0, 0.5, 0.02, 0.15),
        (130.0, 100.0, 2.0, 0.05, 0.40),
        (70.0, 100.0, 0.25, 0.02, 0.15),
    ],
)
def test_price_at_least_intrinsic(S0, K, T, r, sigma):
    contract = _contract(S0=S0, K=K, T=T, r=r, sigma=sigma)
    price = crr_price(contract)
    intrinsic = max(K - S0, 0.0)
    assert price >= intrinsic - 1e-8


@pytest.mark.parametrize(
    "S0,K,T,r,sigma",
    [
        (100.0, 100.0, 1.0, 0.05, 0.25),
        (90.0, 100.0, 0.5, 0.02, 0.20),
        (110.0, 100.0, 2.0, 0.05, 0.30),
    ],
)
def test_american_price_at_least_european(S0, K, T, r, sigma):
    contract = _contract(S0=S0, K=K, T=T, r=r, sigma=sigma)
    american = crr_price(contract)
    european = black_scholes_price(contract)
    assert american >= european - 1e-8


def test_deep_itm_tiny_sigma_converges_to_intrinsic():
    contract = _contract(S0=50.0, K=100.0, T=0.01, r=0.02, sigma=0.01, steps=200)
    price = crr_price(contract)
    intrinsic = max(contract.K - contract.S0, 0.0)
    assert abs(price - intrinsic) < 0.5


def test_monotonic_decrease_in_spot():
    spots = [70.0, 90.0, 110.0, 130.0]
    prices = [crr_price(_contract(S0=s)) for s in spots]
    for earlier, later in zip(prices, prices[1:]):
        assert earlier >= later


def test_arbitrage_violation_raises():
    # u <= exp(r*dt): sigma too small relative to a huge rate over dt.
    contract = _contract(S0=100.0, K=100.0, T=1.0, r=5.0, sigma=0.0001, steps=10)
    with pytest.raises(ValueError):
        crr_price(contract)


def test_exercise_boundary_non_empty_and_sane():
    contract = _contract(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=50)
    price, boundary = crr_american_put_with_boundary(contract)
    assert len(boundary) > 0
    non_none = [v for v in boundary.values() if v is not None]
    assert len(non_none) > 0
    for v in non_none:
        assert 0.0 < v <= contract.K
