import pytest
import numpy as np
from stats_functions import (
    calculate_sample_size,
    calculate_power,
    simulate_experiment,
    calculate_mde,
    calculate_confidence_interval,
    calculate_relative_lift,
    calculate_cuped_adjustment,
    cuped_ab_test,
    calculate_cuped_sample_size,
)


# --- sample size ---

def test_sample_size_basic():
    n = calculate_sample_size(baseline_rate=0.10, mde=0.02)
    assert 3000 < n < 5000

def test_sample_size_smaller_effect_needs_more():
    n_small = calculate_sample_size(baseline_rate=0.10, mde=0.01)
    n_large = calculate_sample_size(baseline_rate=0.10, mde=0.05)
    assert n_small > n_large

def test_sample_size_higher_power():
    n_80 = calculate_sample_size(baseline_rate=0.10, mde=0.02, power=0.80)
    n_90 = calculate_sample_size(baseline_rate=0.10, mde=0.02, power=0.90)
    assert n_90 > n_80

def test_sample_size_returns_int():
    assert isinstance(calculate_sample_size(0.10, 0.02), int)


# --- power ---

def test_power_high_sample():
    power = calculate_power(baseline_rate=0.10, effect_size=0.02, sample_size=50000)
    assert power > 0.95

def test_power_low_sample():
    power = calculate_power(baseline_rate=0.10, effect_size=0.02, sample_size=100)
    assert power < 0.5

def test_power_in_range():
    p = calculate_power(0.10, 0.02, 5000)
    assert 0 < p < 1


def test_mde_decreases_with_sample_size():
    assert calculate_mde(0.10, 1000) > calculate_mde(0.10, 10000)

def test_mde_positive():
    assert calculate_mde(0.10, 5000) > 0


# --- confidence interval ---

def test_ci_contains_estimate():
    rate = 0.12
    lo, hi = calculate_confidence_interval(rate, 5000)
    assert lo < rate < hi

def test_ci_width_shrinks():
    lo_s, hi_s = calculate_confidence_interval(0.12, 100)
    lo_l, hi_l = calculate_confidence_interval(0.12, 10000)
    assert (hi_s - lo_s) > (hi_l - lo_l)

def test_ci_bounds_valid():
    lo, hi = calculate_confidence_interval(0.12, 5000)
    assert 0 <= lo < hi <= 1


def test_lift_20pct():
    assert abs(calculate_relative_lift(0.10, 0.12) - 20.0) < 0.01

def test_lift_zero_baseline():
    assert calculate_relative_lift(0.0, 0.12) == 0.0

def test_lift_negative():
    assert calculate_relative_lift(0.10, 0.08) < 0


# --- simulation ---

def test_sim_keys():
    r = simulate_experiment(0.10, 0.12, 5000, n_simulations=50)
    assert "statistical_power" in r
    assert "p_values" in r

def test_sim_power_ballpark():
    # simulated power should be roughly close to theoretical
    r = simulate_experiment(0.10, 0.12, 5000, n_simulations=500)
    theoretical = calculate_power(0.10, 0.02, 5000)
    assert abs(r["statistical_power"] - theoretical) < 0.10

def test_sim_no_effect():
    r = simulate_experiment(0.10, 0.10, 5000, n_simulations=500)
    # false positive rate should be near alpha (0.05)
    assert r["statistical_power"] < 0.15


# --- CUPED ---

def test_cuped_variance_reduction():
    np.random.seed(42)
    n = 2000
    pre = np.random.normal(0, 1, n)
    post = 0.7 * pre + 0.3 * np.random.normal(0, 1, n)
    _, _, var_red = calculate_cuped_adjustment(post, pre)
    assert var_red > 30

def test_cuped_ab_test_structure():
    np.random.seed(42)
    n = 1000
    ctrl_pre = np.random.normal(0, 1, n)
    ctrl_post = 0.5 * ctrl_pre + np.random.normal(0, 1, n)
    treat_pre = np.random.normal(0, 1, n)
    treat_post = 0.5 * treat_pre + np.random.normal(0.1, 1, n)

    results = cuped_ab_test(ctrl_post, ctrl_pre, treat_post, treat_pre)
    assert "original" in results and "cuped" in results
    # cuped should have smaller SE
    assert results["cuped"]["se"] < results["original"]["se"]

def test_cuped_sample_size_savings():
    original = calculate_sample_size(0.10, 0.02)
    with_cuped = calculate_cuped_sample_size(0.10, 0.02, variance_reduction=0.3)
    assert with_cuped < original
