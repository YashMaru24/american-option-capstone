# American Option Capstone

## Overview

A portfolio-quality capstone comparing three approaches to pricing American
put options: a CRR binomial tree, a neural-network pricer, and a
reinforcement-learning optimal-stopping policy.

## Repo structure

```
american-option-capstone/
  README.md
  requirements.txt
  pyproject.toml
  reports/figures/
  notebooks/exploration.ipynb
  configs/
    nn.yaml
    rl.yaml
  src/
    __init__.py
    contract.py
    pricing/
      __init__.py
      payoffs.py
      black_scholes.py
      binomial.py
      base.py
    data/
      __init__.py
      synthetic_contracts.py
    ml/
      __init__.py
      models.py
      train_nn.py
      evaluate_nn.py
    rl/
      __init__.py
      env.py
      train_dqn.py
      evaluate_policy.py
  tests/
    __init__.py
    test_payoffs.py
    test_binomial.py
    test_nn_sanity.py
    test_rl_env.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

## Status

Parts 1-3 of 4 complete: core CRR binomial pricing, neural-network pricer,
and RL early-exercise policy. Part 4 (cross-method evaluation/reporting) is
still to come.
