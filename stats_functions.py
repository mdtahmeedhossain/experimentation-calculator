"""
Statistical functions for A/B test planning: power analysis, sample size, simulation.
"""

import numpy as np
from scipy import stats
from typing import Dict


def calculate_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True
) -> int:
    """Required sample size per group for a two-proportion z-test."""
    p1 = baseline_rate
    p2 = baseline_rate + mde

    pooled_std = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)

    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    n = 2 * ((z_alpha + z_beta) ** 2) * (pooled_std ** 2) / (mde ** 2)
    return int(np.ceil(n))


def calculate_power(
    baseline_rate: float,
    effect_size: float,
    sample_size: int,
    alpha: float = 0.05,
    two_tailed: bool = True
) -> float:
    """Statistical power for given baseline, effect, and sample size."""
    p1 = baseline_rate
    p2 = baseline_rate + effect_size

    se = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / sample_size)

    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    ncp = effect_size / se
    return 1 - stats.norm.cdf(z_alpha - ncp)


def simulate_experiment(
    baseline_rate: float,
    treatment_rate: float,
    sample_size: int,
    n_simulations: int = 1000,
    alpha: float = 0.05,
    random_seed: int = 42
) -> Dict:
    """Run Monte Carlo simulations of an A/B test to empirically estimate power."""
    np.random.seed(random_seed)

    results = {
        'p_values': [],
        'effect_sizes': [],
        'control_rates': [],
        'treatment_rates': [],
        'significant': []
    }

    for _ in range(n_simulations):
        control = np.random.binomial(1, baseline_rate, sample_size)
        treatment = np.random.binomial(1, treatment_rate, sample_size)

        control_mean = control.mean()
        treatment_mean = treatment.mean()
        effect = treatment_mean - control_mean

        pooled_se = np.sqrt(
            control_mean * (1 - control_mean) / sample_size +
            treatment_mean * (1 - treatment_mean) / sample_size
        )

        if pooled_se > 0:
            z_stat = effect / pooled_se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            p_value = 1.0

        results['p_values'].append(p_value)
        results['effect_sizes'].append(effect)
        results['control_rates'].append(control_mean)
        results['treatment_rates'].append(treatment_mean)
        results['significant'].append(p_value < alpha)

    results['statistical_power'] = np.mean(results['significant'])
    results['mean_effect'] = np.mean(results['effect_sizes'])
    results['std_effect'] = np.std(results['effect_sizes'])

    return results


def calculate_mde(baseline_rate, sample_size, alpha=0.05, power=0.80, two_tailed=True):
    """Minimum detectable effect (absolute) for a given sample size."""
    z_a = stats.norm.ppf(1 - alpha / 2) if two_tailed else stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    p = baseline_rate

    return (z_a + z_b) * np.sqrt(2 * p * (1 - p) / sample_size)


def calculate_confidence_interval(conversion_rate, sample_size, confidence_level=0.95):
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    se = np.sqrt(conversion_rate * (1 - conversion_rate) / sample_size)

    lower = max(0, conversion_rate - z_score * se)
    upper = min(1, conversion_rate + z_score * se)
    return (lower, upper)


def calculate_relative_lift(baseline_rate, treatment_rate):
    if baseline_rate == 0:
        return 0.0
    return ((treatment_rate - baseline_rate) / baseline_rate) * 100

