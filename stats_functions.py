"""
Statistical functions for A/B test planning: power analysis, sample size, simulation, CUPED.
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


# --- CUPED functions ---

def calculate_cuped_adjustment(
    y_post: np.ndarray,
    y_pre: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """
    CUPED adjustment: Y_adj = Y_post - theta * (Y_pre - E[Y_pre])
    where theta = Cov(Y_post, Y_pre) / Var(Y_pre).

    Returns (adjusted_values, theta, variance_reduction_pct).
    """
    cov_matrix = np.cov(y_post, y_pre)
    cov_post_pre = cov_matrix[0, 1]
    var_pre = cov_matrix[1, 1]

    theta = cov_post_pre / var_pre if var_pre > 0 else 0

    y_adjusted = y_post - theta * (y_pre - np.mean(y_pre))

    var_original = np.var(y_post)
    var_adjusted = np.var(y_adjusted)
    variance_reduction_pct = (1 - var_adjusted / var_original) * 100 if var_original > 0 else 0

    return y_adjusted, theta, variance_reduction_pct


def cuped_ab_test(
    control_post: np.ndarray,
    control_pre: np.ndarray,
    treatment_post: np.ndarray,
    treatment_pre: np.ndarray,
    alpha: float = 0.05
) -> Dict:
    """
    Run an A/B test with and without CUPED adjustment.
    Returns both sets of results plus variance reduction stats.
    """
    # Original analysis
    control_mean_orig = np.mean(control_post)
    treatment_mean_orig = np.mean(treatment_post)
    effect_orig = treatment_mean_orig - control_mean_orig

    n_control = len(control_post)
    n_treatment = len(treatment_post)
    se_orig = np.sqrt(
        np.var(control_post) / n_control +
        np.var(treatment_post) / n_treatment
    )

    # CUPED-adjusted
    control_adjusted, theta_control, var_red_control = calculate_cuped_adjustment(
        control_post, control_pre
    )
    treatment_adjusted, theta_treatment, var_red_treatment = calculate_cuped_adjustment(
        treatment_post, treatment_pre
    )

    control_mean_cuped = np.mean(control_adjusted)
    treatment_mean_cuped = np.mean(treatment_adjusted)
    effect_cuped = treatment_mean_cuped - control_mean_cuped

    se_cuped = np.sqrt(
        np.var(control_adjusted) / n_control +
        np.var(treatment_adjusted) / n_treatment
    )

    # Z-tests
    if se_orig > 0:
        z_stat_orig = effect_orig / se_orig
        p_value_orig = 2 * (1 - stats.norm.cdf(abs(z_stat_orig)))
    else:
        z_stat_orig, p_value_orig = 0, 1.0

    if se_cuped > 0:
        z_stat_cuped = effect_cuped / se_cuped
        p_value_cuped = 2 * (1 - stats.norm.cdf(abs(z_stat_cuped)))
    else:
        z_stat_cuped, p_value_cuped = 0, 1.0

    z_critical = stats.norm.ppf(1 - alpha / 2)

    ci_orig = (
        effect_orig - z_critical * se_orig,
        effect_orig + z_critical * se_orig
    )
    ci_cuped = (
        effect_cuped - z_critical * se_cuped,
        effect_cuped + z_critical * se_cuped
    )

    avg_var_reduction = (var_red_control + var_red_treatment) / 2

    all_pre = np.concatenate([control_pre, treatment_pre])
    all_post = np.concatenate([control_post, treatment_post])
    correlation = np.corrcoef(all_pre, all_post)[0, 1]

    return {
        'original': {
            'control_mean': control_mean_orig,
            'treatment_mean': treatment_mean_orig,
            'effect': effect_orig,
            'se': se_orig,
            'z_stat': z_stat_orig,
            'p_value': p_value_orig,
            'ci': ci_orig,
            'significant': p_value_orig < alpha
        },
        'cuped': {
            'control_mean': control_mean_cuped,
            'treatment_mean': treatment_mean_cuped,
            'effect': effect_cuped,
            'se': se_cuped,
            'z_stat': z_stat_cuped,
            'p_value': p_value_cuped,
            'ci': ci_cuped,
            'significant': p_value_cuped < alpha
        },
        'variance_reduction': {
            'control_pct': var_red_control,
            'treatment_pct': var_red_treatment,
            'average_pct': avg_var_reduction,
            'theta_control': theta_control,
            'theta_treatment': theta_treatment,
            'correlation': correlation
        },
        'sample_sizes': {
            'control': n_control,
            'treatment': n_treatment
        }
    }


def calculate_experiment_duration(sample_size_per_group, daily_visitors, traffic_allocation=1.0):
    """Estimate how many days/weeks to run an experiment given traffic."""
    total_needed = sample_size_per_group * 2
    effective_daily = daily_visitors * traffic_allocation

    if effective_daily <= 0:
        return {"error": "Daily visitors must be positive"}

    days = total_needed / effective_daily
    return {
        "sample_size_per_group": sample_size_per_group,
        "total_sample_needed": total_needed,
        "daily_visitors": daily_visitors,
        "traffic_allocation_pct": traffic_allocation * 100,
        "effective_daily_visitors": effective_daily,
        "days_needed": round(days, 1),
        "weeks_needed": round(days / 7, 1)
    }


def calculate_cuped_sample_size(baseline_rate, mde, variance_reduction,
                                alpha=0.05, power=0.80, two_tailed=True):
    # adjusted sample size accounting for CUPED variance reduction
    orig = calculate_sample_size(baseline_rate, mde, alpha, power, two_tailed)
    return int(np.ceil(orig * (1 - variance_reduction)))
