"""
A/B Testing Experimentation Calculator
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from stats_functions import (
    calculate_sample_size,
    calculate_power,
    simulate_experiment,
    calculate_mde,
    calculate_confidence_interval,
    calculate_relative_lift,
    cuped_ab_test,
    calculate_cuped_adjustment,
    calculate_cuped_sample_size
)

st.set_page_config(
    page_title="A/B Testing Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">A/B Testing Experimentation Calculator</p>', unsafe_allow_html=True)
st.markdown("**Plan, analyze, and validate your experiments with statistical rigor**")
st.markdown("---")

with st.sidebar:
    st.header("About")
    st.markdown("""
    This tool helps you:
    - Calculate required sample sizes
    - Estimate statistical power
    - Simulate experiment outcomes
    - Make data-driven decisions

    **Created by:** Tahmeed Hossain
    **GitHub:** [mdtahmeedhossain](https://github.com/mdtahmeedhossain)
    """)

    st.markdown("---")
    st.header("Global Settings")

    alpha = st.select_slider(
        "Significance Level (α)",
        options=[0.01, 0.05, 0.10],
        value=0.05,
        help="Probability of Type I error (false positive)"
    )

    confidence_level = 1 - alpha
    st.info(f"Confidence Level: {confidence_level*100:.0f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Sample Size Calculator",
    "Power Analysis",
    "Experiment Simulator",
    "CUPED Analysis",
    "AI Assistant"
])

with tab1:
    st.header("Sample Size Calculator")
    st.markdown("Determine how many users you need to detect a meaningful effect.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Parameters")

        baseline_rate_ss = st.slider(
            "Baseline Conversion Rate (%)",
            min_value=1.0,
            max_value=50.0,
            value=10.0,
            step=0.5
        ) / 100

        mde_type = st.radio(
            "Specify MDE as:",
            ["Absolute Change", "Relative Change"]
        )

        if mde_type == "Absolute Change":
            mde_absolute = st.slider(
                "Minimum Detectable Effect (percentage points)",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5,
                help="Absolute difference you want to detect (e.g., 10% -> 12% is 2pp)"
            ) / 100
            treatment_rate_ss = baseline_rate_ss + mde_absolute
        else:
            relative_change = st.slider(
                "Relative Improvement (%)",
                min_value=5.0,
                max_value=50.0,
                value=20.0,
                step=5.0,
                help="Percentage improvement over baseline (e.g., 20% means 10% -> 12%)"
            ) / 100
            treatment_rate_ss = baseline_rate_ss * (1 + relative_change)
            mde_absolute = treatment_rate_ss - baseline_rate_ss

        power_ss = st.slider(
            "Statistical Power",
            min_value=0.70,
            max_value=0.95,
            value=0.80,
            step=0.05,
            help="Probability of detecting the effect if it exists (1 - β)"
        )

        st.info(f"""
        **Target Rates:**
        - Control: {baseline_rate_ss*100:.2f}%
        - Treatment: {treatment_rate_ss*100:.2f}%
        - Absolute Difference: {mde_absolute*100:.2f}pp
        - Relative Lift: {calculate_relative_lift(baseline_rate_ss, treatment_rate_ss):.1f}%
        """)

    with col2:
        st.subheader("Results")

        sample_size_required = calculate_sample_size(
            baseline_rate_ss,
            mde_absolute,
            alpha=alpha,
            power=power_ss
        )

        st.metric(
            "Sample Size per Group",
            f"{sample_size_required:,}"
        )

        st.metric(
            "Total Sample Size",
            f"{sample_size_required * 2:,}"
        )

        st.markdown("### Experiment Duration Estimator")
        daily_traffic = st.number_input(
            "Daily Traffic (users/day)",
            min_value=100,
            max_value=1000000,
            value=10000,
            step=1000
        )

        days_required = (sample_size_required * 2) / daily_traffic
        weeks_required = days_required / 7

        col_a, col_b = st.columns(2)
        col_a.metric("Days Required", f"{days_required:.1f}")
        col_b.metric("Weeks Required", f"{weeks_required:.1f}")

        if days_required < 7:
            st.success("Experiment can be completed in less than a week!")
        elif days_required < 14:
            st.info("Experiment will take 1-2 weeks")
        else:
            st.warning("Long experiment duration - consider increasing MDE or reducing power")

    st.divider()
    st.subheader("Sensitivity Analysis: Sample Size vs Effect Size")

    effect_sizes = np.linspace(0.005, 0.05, 20)
    sample_sizes = [
        calculate_sample_size(baseline_rate_ss, es, alpha=alpha, power=power_ss)
        for es in effect_sizes
    ]

    fig_sensitivity = go.Figure()
    fig_sensitivity.add_trace(go.Scatter(
        x=effect_sizes * 100,
        y=sample_sizes,
        mode='lines+markers',
        name='Sample Size',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))

    fig_sensitivity.update_layout(
        title="How Sample Size Changes with Effect Size",
        xaxis_title="Minimum Detectable Effect (percentage points)",
        yaxis_title="Sample Size per Group",
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig_sensitivity, use_container_width=True)

with tab2:
    st.header("Statistical Power Analysis")
    st.write("Estimate the power of your experiment given available sample size.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Parameters")

        baseline_rate_power = st.slider(
            "Baseline Conversion Rate (%)",
            min_value=1.0,
            max_value=50.0,
            value=10.0,
            step=0.5,
            key="power_baseline"
        ) / 100

        effect_size_power = st.slider(
            "Expected Effect Size (percentage points)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="power_effect"
        ) / 100

        sample_size_power = st.number_input(
            "Available Sample Size per Group",
            min_value=100,
            max_value=1000000,
            value=5000,
            step=500,
            key="power_sample"
        )

        treatment_rate_power = baseline_rate_power + effect_size_power

        st.info(f"""
        **Scenario:**
        - Control: {baseline_rate_power*100:.2f}%
        - Treatment: {treatment_rate_power*100:.2f}%
        - Relative Lift: {calculate_relative_lift(baseline_rate_power, treatment_rate_power):.1f}%
        """)

    with col2:
        st.subheader("Results")

        power_achieved = calculate_power(
            baseline_rate_power,
            effect_size_power,
            sample_size_power,
            alpha=alpha
        )

        st.metric(
            "Statistical Power",
            f"{power_achieved*100:.1f}%"
        )

        if power_achieved >= 0.80:
            st.success("Your experiment has sufficient power (>= 80%)")
        elif power_achieved >= 0.70:
            st.info("Moderate power. Consider increasing sample size if possible.")
        else:
            st.error("Low power. High risk of missing a real effect!")

        mde_achievable = calculate_mde(
            baseline_rate_power,
            sample_size_power,
            alpha=alpha,
            power=0.80
        )

        st.markdown("### Minimum Detectable Effect")
        st.metric(
            "MDE at 80% Power",
            f"{mde_achievable*100:.2f}pp",
            help="Smallest effect you can reliably detect with this sample size"
        )

        st.info(f"""
        With {sample_size_power:,} users per group, you can detect:
        - Absolute change: {mde_achievable*100:.2f} percentage points
        - Relative change: {(mde_achievable/baseline_rate_power)*100:.1f}%
        """)

    st.markdown("---")
    st.subheader("Power Curves: Effect Size vs Power")

    effect_range = np.linspace(0.005, 0.05, 30)
    power_values = [
        calculate_power(baseline_rate_power, es, sample_size_power, alpha=alpha)
        for es in effect_range
    ]

    fig_power = go.Figure()

    fig_power.add_trace(go.Scatter(
        x=effect_range * 100,
        y=power_values,
        mode='lines',
        name=f'N={sample_size_power:,}',
        line=dict(color='#1f77b4', width=3)
    ))

    fig_power.add_hline(y=0.80, line_dash="dash", line_color="green",
                        annotation_text="80% Power (recommended)")
    fig_power.add_vline(x=effect_size_power * 100, line_dash="dash", line_color="red",
                        annotation_text="Your Effect Size")

    fig_power.update_layout(
        title="Statistical Power for Different Effect Sizes",
        xaxis_title="Effect Size (percentage points)",
        yaxis_title="Statistical Power",
        yaxis_range=[0, 1],
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig_power, use_container_width=True)

    st.markdown("### Compare Multiple Sample Sizes")

    sample_sizes_compare = [
        int(sample_size_power * 0.5),
        sample_size_power,
        int(sample_size_power * 1.5),
        int(sample_size_power * 2)
    ]

    fig_compare = go.Figure()

    for n in sample_sizes_compare:
        power_vals = [
            calculate_power(baseline_rate_power, es, n, alpha=alpha)
            for es in effect_range
        ]
        fig_compare.add_trace(go.Scatter(
            x=effect_range * 100,
            y=power_vals,
            mode='lines',
            name=f'N={n:,}',
            line=dict(width=2)
        ))

    fig_compare.add_hline(y=0.80, line_dash="dash", line_color="gray")

    fig_compare.update_layout(
        title="Power Curves for Different Sample Sizes",
        xaxis_title="Effect Size (percentage points)",
        yaxis_title="Statistical Power",
        yaxis_range=[0, 1],
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig_compare, use_container_width=True)

with tab3:
    st.header("Experiment Simulator")
    st.write("Monte Carlo simulation to empirically validate your power calculations.")

    sim_col1, sim_col2, sim_col3 = st.columns(3)

    with sim_col1:
        baseline_rate_sim = st.slider(
            "Control Rate (%)", 1.0, 50.0, 10.0, 0.5, key="sim_baseline"
        ) / 100
        treatment_rate_sim = st.slider(
            "Treatment Rate (%)", 1.0, 50.0, 12.0, 0.5, key="sim_treatment"
        ) / 100

    with sim_col2:
        sample_size_sim = st.number_input(
            "Sample Size per Group", 100, 100000, 5000, 500, key="sim_sample"
        )
        n_simulations = st.select_slider(
            "Number of Simulations",
            options=[100, 500, 1000, 2000, 5000],
            value=1000
        )

    true_effect = treatment_rate_sim - baseline_rate_sim
    relative_lift_sim = calculate_relative_lift(baseline_rate_sim, treatment_rate_sim)

    expected_power = calculate_power(
        baseline_rate_sim, true_effect, sample_size_sim, alpha=alpha
    )

    with sim_col3:
        st.metric("Expected Power", f"{expected_power*100:.1f}%")
        st.metric("True Effect", f"{true_effect*100:.2f}pp ({relative_lift_sim:.1f}% relative)")

    run_simulation = st.button("Run Simulation", type="primary", use_container_width=True)

    if run_simulation:
        with st.spinner(f"Running {n_simulations} simulated experiments..."):
            results = simulate_experiment(
                baseline_rate_sim,
                treatment_rate_sim,
                sample_size_sim,
                n_simulations=n_simulations,
                alpha=alpha
            )

        st.success("Simulation complete!")

        st.markdown("---")
        st.subheader("Simulation Results")

        col_a, col_b, col_c, col_d = st.columns(4)

        col_a.metric(
            "Observed Power",
            f"{results['statistical_power']*100:.1f}%",
            delta=f"{(results['statistical_power'] - expected_power)*100:.1f}pp"
        )

        col_b.metric(
            "Significant Results",
            f"{sum(results['significant'])}/{n_simulations}"
        )

        col_c.metric(
            "Mean Effect Size",
            f"{results['mean_effect']*100:.2f}pp",
            delta=f"{(results['mean_effect'] - true_effect)*100:.2f}pp"
        )

        col_d.metric(
            "Std Dev of Effect",
            f"{results['std_effect']*100:.2f}pp"
        )

        st.markdown("### Distribution of Results")

        col_viz1, col_viz2 = st.columns(2)

        with col_viz1:
            fig_pval = go.Figure()
            fig_pval.add_trace(go.Histogram(
                x=results['p_values'],
                nbinsx=50,
                name='P-values',
                marker_color='#1f77b4'
            ))
            fig_pval.add_vline(x=alpha, line_dash="dash", line_color="red",
                              annotation_text=f"α={alpha}")
            fig_pval.update_layout(
                title="Distribution of P-values",
                xaxis_title="P-value",
                yaxis_title="Frequency",
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig_pval, use_container_width=True)

        with col_viz2:
            fig_effect = go.Figure()
            fig_effect.add_trace(go.Histogram(
                x=[e * 100 for e in results['effect_sizes']],
                nbinsx=50,
                name='Effect Sizes',
                marker_color='#2ca02c'
            ))
            fig_effect.add_vline(x=true_effect * 100, line_dash="dash", line_color="red",
                                annotation_text="True Effect")
            fig_effect.update_layout(
                title="Distribution of Observed Effect Sizes",
                xaxis_title="Effect Size (percentage points)",
                yaxis_title="Frequency",
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig_effect, use_container_width=True)

        st.markdown("### Observed Conversion Rates")

        fig_scatter = go.Figure()

        colors = ['green' if sig else 'red' for sig in results['significant']]

        fig_scatter.add_trace(go.Scatter(
            x=[r * 100 for r in results['control_rates']],
            y=[r * 100 for r in results['treatment_rates']],
            mode='markers',
            marker=dict(
                color=colors,
                size=5,
                opacity=0.5
            ),
            text=[f"p={p:.4f}" for p in results['p_values']],
            hovertemplate="Control: %{x:.2f}%<br>Treatment: %{y:.2f}%<br>%{text}<extra></extra>"
        ))

        min_rate = min(min(results['control_rates']), min(results['treatment_rates'])) * 100
        max_rate = max(max(results['control_rates']), max(results['treatment_rates'])) * 100
        fig_scatter.add_trace(go.Scatter(
            x=[min_rate, max_rate],
            y=[min_rate, max_rate],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='No Effect Line',
            showlegend=True
        ))

        fig_scatter.update_layout(
            title="Observed Conversion Rates (Green = Significant, Red = Not Significant)",
            xaxis_title="Control Group Rate (%)",
            yaxis_title="Treatment Group Rate (%)",
            height=400,
            showlegend=True
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("### Interpretation")

        if abs(results['statistical_power'] - expected_power) < 0.05:
            st.success(f"Observed power ({results['statistical_power']*100:.1f}%) matches theoretical ({expected_power*100:.1f}%) - design looks good.")
        else:
            st.warning(f"Observed power ({results['statistical_power']*100:.1f}%) differs from theoretical ({expected_power*100:.1f}%). Try running more simulations.")

with tab4:
    st.header("CUPED Variance Reduction Analysis")
    st.markdown("Use pre-experiment covariates to reduce variance and detect smaller effects with fewer samples.")

    st.markdown("---")

    data_input_method = st.radio(
        "Data Input Method",
        ["Simulate Data", "Manual Input"]
    )

    if data_input_method == "Simulate Data":
        st.subheader("Simulate Experiment with CUPED")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Experiment Parameters**")

            baseline_cuped = st.slider(
                "Baseline Conversion Rate (%)",
                min_value=1.0,
                max_value=50.0,
                value=10.0,
                step=0.5,
                key="cuped_baseline"
            )

            treatment_rate_cuped = st.slider(
                "Treatment Conversion Rate (%)",
                min_value=1.0,
                max_value=50.0,
                value=12.0,
                step=0.5,
                key="cuped_treatment"
            )

            sample_size_cuped = st.number_input(
                "Sample Size per Group",
                min_value=100,
                max_value=100000,
                value=5000,
                step=100,
                key="cuped_sample_size"
            )

        with col2:
            st.markdown("**Pre-Experiment Covariate**")

            correlation_strength = st.slider(
                "Correlation with Outcome (r)",
                min_value=0.0,
                max_value=0.9,
                value=0.6,
                step=0.05,
                help="How strongly the pre-experiment metric correlates with the outcome. Higher = more variance reduction."
            )

            st.info(f"""
            **Expected Variance Reduction:** ~{(correlation_strength**2)*100:.0f}%

            With r={correlation_strength:.2f}, CUPED can reduce variance by approximately {(correlation_strength**2)*100:.0f}%.
            """)

        if st.button("Run CUPED Analysis", key="run_cuped"):
            with st.spinner("Running analysis..."):
                p_control = baseline_cuped / 100
                p_treatment = treatment_rate_cuped / 100
                n = int(sample_size_cuped)

                np.random.seed(42)

                control_pre = np.random.binomial(1, p_control, n).astype(float)
                treatment_pre = np.random.binomial(1, p_treatment, n).astype(float)

                # generate post data with correlation to pre data
                control_post = (
                    correlation_strength * control_pre +
                    (1 - correlation_strength) * np.random.binomial(1, p_control, n)
                ).clip(0, 1)

                treatment_post = (
                    correlation_strength * treatment_pre +
                    (1 - correlation_strength) * np.random.binomial(1, p_treatment, n)
                ).clip(0, 1)

                results = cuped_ab_test(
                    control_post, control_pre,
                    treatment_post, treatment_pre,
                    alpha=alpha
                )

                st.markdown("---")
                st.subheader("Results Comparison")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Variance Reduction",
                        f"{results['variance_reduction']['average_pct']:.1f}%"
                    )

                with col2:
                    st.metric(
                        "Correlation (Pre-Post)",
                        f"{results['variance_reduction']['correlation']:.3f}"
                    )

                with col3:
                    improvement = (
                        (results['original']['se'] - results['cuped']['se']) /
                        results['original']['se'] * 100
                    )
                    st.metric(
                        "Standard Error Reduction",
                        f"{improvement:.1f}%"
                    )

                st.markdown("### Original vs CUPED-Adjusted Analysis")

                comparison_df = pd.DataFrame({
                    'Metric': [
                        'Control Mean',
                        'Treatment Mean',
                        'Effect Size',
                        'Standard Error',
                        'Z-Statistic',
                        'P-Value',
                        'Significant?',
                        'CI Lower',
                        'CI Upper'
                    ],
                    'Original': [
                        f"{results['original']['control_mean']:.4f}",
                        f"{results['original']['treatment_mean']:.4f}",
                        f"{results['original']['effect']:.4f}",
                        f"{results['original']['se']:.4f}",
                        f"{results['original']['z_stat']:.3f}",
                        f"{results['original']['p_value']:.4f}",
                        'Yes' if results['original']['significant'] else 'No',
                        f"{results['original']['ci'][0]:.4f}",
                        f"{results['original']['ci'][1]:.4f}"
                    ],
                    'CUPED-Adjusted': [
                        f"{results['cuped']['control_mean']:.4f}",
                        f"{results['cuped']['treatment_mean']:.4f}",
                        f"{results['cuped']['effect']:.4f}",
                        f"{results['cuped']['se']:.4f}",
                        f"{results['cuped']['z_stat']:.3f}",
                        f"{results['cuped']['p_value']:.4f}",
                        'Yes' if results['cuped']['significant'] else 'No',
                        f"{results['cuped']['ci'][0]:.4f}",
                        f"{results['cuped']['ci'][1]:.4f}"
                    ]
                })

                st.dataframe(comparison_df, use_container_width=True)

                st.markdown("### Confidence Interval Comparison")

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=[results['original']['effect']],
                    y=['Original'],
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=[results['original']['ci'][1] - results['original']['effect']],
                        arrayminus=[results['original']['effect'] - results['original']['ci'][0]]
                    ),
                    mode='markers',
                    marker=dict(size=12, color='#ff7f0e'),
                    name='Original'
                ))

                fig.add_trace(go.Scatter(
                    x=[results['cuped']['effect']],
                    y=['CUPED'],
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=[results['cuped']['ci'][1] - results['cuped']['effect']],
                        arrayminus=[results['cuped']['effect'] - results['cuped']['ci'][0]]
                    ),
                    mode='markers',
                    marker=dict(size=12, color='#2ca02c'),
                    name='CUPED'
                ))

                fig.add_vline(x=0, line_dash="dash", line_color="gray")

                fig.update_layout(
                    title="95% Confidence Intervals: Original vs CUPED",
                    xaxis_title="Effect Size",
                    yaxis_title="Method",
                    height=300,
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Sample Size Savings")

                var_reduction = results['variance_reduction']['average_pct'] / 100
                original_n = int(sample_size_cuped)
                cuped_n = int(original_n * (1 - var_reduction))
                savings = original_n - cuped_n

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Original Sample Size Needed",
                        f"{original_n:,}"
                    )

                with col2:
                    st.metric(
                        "With CUPED",
                        f"{cuped_n:,}",
                        delta=f"-{savings:,}",
                        delta_color="normal"
                    )

                with col3:
                    st.metric(
                        "Savings",
                        f"{(savings/original_n)*100:.1f}%"
                    )

                st.markdown("---")
                st.subheader("Interpretation")

                if results['variance_reduction']['average_pct'] > 20:
                    st.success(f"Variance reduced by {results['variance_reduction']['average_pct']:.1f}% - you could use ~{(savings/original_n)*100:.0f}% fewer users for the same power.")
                elif results['variance_reduction']['average_pct'] > 10:
                    st.info(f"Variance reduced by {results['variance_reduction']['average_pct']:.1f}% - moderate improvement.")
                else:
                    st.warning(f"Only {results['variance_reduction']['average_pct']:.1f}% variance reduction. The covariate may be too weakly correlated - try a different one.")

    else:  # Manual Input
        st.subheader("Manual Data Input")
        st.info("Enter pre-experiment and post-experiment values for both groups.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Control Group**")
            control_pre_input = st.text_area(
                "Pre-Experiment Values (comma-separated)",
                "0.08, 0.12, 0.09, 0.11, 0.10",
                key="control_pre"
            )
            control_post_input = st.text_area(
                "Post-Experiment Values (comma-separated)",
                "0.09, 0.13, 0.08, 0.12, 0.11",
                key="control_post"
            )

        with col2:
            st.markdown("**Treatment Group**")
            treatment_pre_input = st.text_area(
                "Pre-Experiment Values (comma-separated)",
                "0.09, 0.11, 0.10, 0.12, 0.08",
                key="treatment_pre"
            )
            treatment_post_input = st.text_area(
                "Post-Experiment Values (comma-separated)",
                "0.11, 0.14, 0.12, 0.15, 0.10",
                key="treatment_post"
            )

        if st.button("Analyze with CUPED", key="analyze_manual"):
            try:
                control_pre = np.array([float(x.strip()) for x in control_pre_input.split(',')])
                control_post = np.array([float(x.strip()) for x in control_post_input.split(',')])
                treatment_pre = np.array([float(x.strip()) for x in treatment_pre_input.split(',')])
                treatment_post = np.array([float(x.strip()) for x in treatment_post_input.split(',')])

                if len(control_pre) != len(control_post):
                    st.error("Control group: pre and post arrays must have the same length!")
                elif len(treatment_pre) != len(treatment_post):
                    st.error("Treatment group: pre and post arrays must have the same length!")
                else:
                    results = cuped_ab_test(
                        control_post, control_pre,
                        treatment_post, treatment_pre,
                        alpha=alpha
                    )

                    st.success(f"Analysis complete! Analyzed {len(control_pre)} control and {len(treatment_pre)} treatment observations.")

                    st.markdown("---")
                    st.subheader("Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Variance Reduction",
                            f"{results['variance_reduction']['average_pct']:.1f}%"
                        )

                    with col2:
                        st.metric(
                            "Correlation (Pre-Post)",
                            f"{results['variance_reduction']['correlation']:.3f}"
                        )

                    comparison_df = pd.DataFrame({
                        'Metric': ['Effect Size', 'Standard Error', 'P-Value', 'Significant?'],
                        'Original': [
                            f"{results['original']['effect']:.4f}",
                            f"{results['original']['se']:.4f}",
                            f"{results['original']['p_value']:.4f}",
                            'Yes' if results['original']['significant'] else 'No'
                        ],
                        'CUPED': [
                            f"{results['cuped']['effect']:.4f}",
                            f"{results['cuped']['se']:.4f}",
                            f"{results['cuped']['p_value']:.4f}",
                            'Yes' if results['cuped']['significant'] else 'No'
                        ]
                    })

                    st.dataframe(comparison_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing input: {str(e)}")
                st.info("Make sure all values are numbers separated by commas.")

with tab5:
    st.header("AI Experiment Design Assistant")
    st.markdown("""
    Describe your experiment scenario in plain English, and I'll help you calculate sample sizes,
    power, and duration estimates.

    **Example questions:**
    - *"Our checkout conversion is 3.2%. We want to detect a 10% relative improvement. How many users do we need?"*
    - *"We have 5,000 daily users. How long would it take to run this experiment?"*
    - *"Can we use CUPED to reduce the sample size? We have historical purchase data."*
    """)

    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.warning("OpenAI API key not found. Please add your API key to a `.env` file:")
        st.code("OPENAI_API_KEY=sk-your-key-here", language="bash")
        st.info("Once you add the key, restart the Streamlit app.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "assistant" not in st.session_state:
            try:
                from llm_assistant import ExperimentAssistant
                st.session_state.assistant = ExperimentAssistant(api_key=api_key)
            except Exception as e:
                st.error(f"Failed to initialize assistant: {e}")
                st.session_state.assistant = None

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Describe your experiment scenario..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if st.session_state.assistant:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.assistant.chat(prompt)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                if st.session_state.assistant:
                    st.session_state.assistant.reset_conversation()
                st.rerun()

        st.markdown("---")
        st.markdown("**Quick examples** (click to try):")

        example_prompts = [
            "Our homepage has a 5% click-through rate. We want to test a new hero banner and detect at least a 15% relative improvement. What sample size do we need?",
            "We have 2,000 daily active users. Our current purchase rate is 2.5%. If we expect a 0.5 percentage point lift, how long should we run the test?",
            "What's the minimum effect we can detect with 10,000 users per group if our baseline is 8%?"
        ]

        for i, example in enumerate(example_prompts):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state.messages.append({"role": "user", "content": example})
                if st.session_state.assistant:
                    response = st.session_state.assistant.chat(example)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit | <a href='https://github.com/mdtahmeedhossain'>Tahmeed Hossain</a></p>
</div>
""", unsafe_allow_html=True)
