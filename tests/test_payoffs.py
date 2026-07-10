import numpy as np
import pytest

from src.pricing.payoffs import call_payoff, payoff, put_payoff


def test_put_payoff_zero_when_itm_reversed():
    assert put_payoff(150.0, 100.0) == 0.0


def test_put_payoff_positive_itm():
    assert put_payoff(80.0, 100.0) == 20.0


def test_call_payoff_zero_when_otm():
    assert call_payoff(80.0, 100.0) == 0.0


def test_call_payoff_positive_itm():
    assert call_payoff(120.0, 100.0) == 20.0


def test_put_payoff_at_the_money():
    assert put_payoff(100.0, 100.0) == 0.0


def test_call_payoff_at_the_money():
    assert call_payoff(100.0, 100.0) == 0.0


def test_put_payoff_vectorized():
    S = np.array([70.0, 100.0, 130.0])
    result = put_payoff(S, 100.0)
    expected = np.array([30.0, 0.0, 0.0])
    np.testing.assert_allclose(result, expected)


def test_call_payoff_vectorized():
    S = np.array([70.0, 100.0, 130.0])
    result = call_payoff(S, 100.0)
    expected = np.array([0.0, 0.0, 30.0])
    np.testing.assert_allclose(result, expected)


def test_payoff_dispatch_put():
    assert payoff(80.0, 100.0, "put") == 20.0


def test_payoff_dispatch_call():
    assert payoff(120.0, 100.0, "call") == 20.0


def test_payoff_invalid_option_type_raises():
    with pytest.raises(ValueError):
        payoff(100.0, 100.0, "straddle")


def test_negative_spot_put_payoff_still_correct():
    # Not economically meaningful but should not crash and follow formula.
    assert put_payoff(-10.0, 100.0) == 110.0
