import seaborn as sns
import itertools
from scipy.stats import mannwhitneyu, wilcoxon, t, sem
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy

def ci_95(data):
    '''
    function to calculate the 95% confidence interval of the mean using the standard CI formula, to be used in
    a sns.pointplot()
    :param data: input data
    :return: 95% CI to be used as error bars in sns.pointplot
    '''
    n = len(data)
    mean = np.mean(data)
    sem_ = sem(data)  # Standard error of the mean
    ci = sem_ * t.ppf(0.975, n - 1)  # 95% CI using t-distribution

    return mean - ci, mean + ci

def assess_differences_mean_ci(df):
    '''
    Function to test whether performance differs across algorithms and feature sets.
    Uses the Wilcoxon rank-sum test for statistical comparisons and calculates confidence intervals.
    Two figures are created: one with and one without horizontal lines indicating statistical differences (p<0.05).
    :param df: DataFrame containing performance metrics (e.g., auc, sensitivity).
    :param metrics: List of metrics to analyze (default: ['auc', 'sensitivity']).
    :return: Dictionary containing statistical comparison results for each metric.
    '''

    # Rename feature sets for better readability
    df.loc[df['feature_set'] == 'emr', 'feature_set'] = 'EMR'
    df.loc[df['feature_set'] == 'emr_sensor', 'feature_set'] = 'EMR+sensor'
    df.loc[df['feature_set'] == 'sensor', 'feature_set'] = 'Sensor'

    # Initialize a dictionary to store results for each metric
    metric_results = {}

    # Iterate over the specified metrics
    for metric in ['auc', 'sensitivity']:

        ### APPLY STATISTICAL TESTS
        # Group data by algorithm and feature set
        grouped = df.groupby(['algorithm', 'feature_set'])
        unique_groups = grouped.groups.keys()

        # Initialize a DataFrame to store pairwise comparisons
        comparison_df = pd.DataFrame()

        # Perform pairwise comparisons between all unique group combinations
        for (ml1, hd1), (ml2, hd2) in itertools.combinations(unique_groups, 2):

            # Skip comparisons between different algorithms
            if ml1 != ml2:
                continue

            # Extract metric values for the two groups being compared
            group1 = grouped.get_group((ml1, hd1))[metric]
            group2 = grouped.get_group((ml2, hd2))[metric]

            # Perform the Wilcoxon rank-sum test
            try:
                _, p_val = wilcoxon(group1, group2, alternative='two-sided')
            except:
                p_val = np.nan

            # Calculate confidence intervals and mean for group 1
            lower_bound_1, upper_bound_1 = ci_95(group1)
            mean_1 = np.mean(group1)

            # Calculate confidence intervals and mean for group 2
            lower_bound_2, upper_bound_2 = ci_95(group2)
            mean_2 = np.mean(group2)

            # Store the results in the comparison DataFrame
            comparison_df = pd.concat([comparison_df, pd.DataFrame({
                'algorithm_1': ml1, 'feature_set_1': hd1,
                'mean_1': mean_1, 'ci_low_1': lower_bound_1, 'ci_high_1': upper_bound_1,
                'algorithm_2': ml2, 'feature_set_2': hd2,
                'mean_2': mean_2, 'ci_low_2': lower_bound_2, 'ci_high_2': upper_bound_2,
                'p': p_val, 'alternative': 'two-sided'
            }, index=[0])], ignore_index=True)

        # Store the comparison results for the current metric
        metric_results[metric] = comparison_df

    # Return the dictionary containing results for all metrics
    return metric_results

def sensor_stats_correlation(df_pred, df_imp):
    '''
    Analyzes the correlation of most important sensor features in a machine learning model by calculating the interquartile range (IQR)
    and median absolute correlation coefficients with electronic medical record (EMR) features.
    :param df_pred: DataFrame with all predictions for an algorithm.
    :param df_imp: DataFrame with all feature importances for an algorithm.
    :param path: String specifying the file path where the figure should be saved.
    :return: DataFrame with interquartile ranges and medians for absolute correlation coefficients with EMR features.
    '''

    # Get lists of different types of features
    sensor_stats_features = [col for col in df_pred.columns if any(feature in col for feature in
                                                                   ['HR', 'RR', 'Act', 'Pos_'])]
    nn_features = [col for col in df_pred.columns if col.startswith("nn_feature_")]
    emr_features = [col for col in df_pred.columns if col not in sensor_stats_features and col not in nn_features and
                    col not in ['y', 'fold', 'repeat', 'subject', 'session', 'y_pred']]

    # Check if neural network features exist
    check_nn = len(nn_features) > 0

    # Initialize DataFrames to store correlations and feature importances
    cor_df_stats= pd.DataFrame()
    cor_df_neural = pd.DataFrame()
    feature_imp = pd.DataFrame()

    # get top 10 statistical features in terms of model contribution
    top_10_sensor_stats_features = (df_imp[sensor_stats_features].fillna(0).median().sort_values(ascending=False)
                                    .head(10))

    # Iterate over unique combinations of repeat and fold
    for repeat in df_pred['repeat'].unique():
        for fold in df_pred['fold'].unique():

            # Initialize a DataFrame to store correlation coefficients for the current iteration
            correlation_df = pd.DataFrame(index=[0])

            # Filter feature importance data for the current repeat and fold
            df_ = df_imp.loc[(df_imp['fold'] == fold) & (df_imp['repeat'] == repeat), :]
            df_ = df_.drop(columns=['fold', 'repeat']).reset_index(drop=True)
            
            # Ensure all neural features exist as columns in the DataFrame
            for feature in nn_features:
                if feature not in df_.columns:
                    df_[feature] = 0  # Add missing column and fill with 0

            # Sort neural network features by importance
            X_values_sorted = df_.loc[0, nn_features].fillna(0).sort_values(ascending=False)
            sorted_nn_features = X_values_sorted.index
            df_[nn_features] = X_values_sorted
            feature_imp = pd.concat([feature_imp, df_], ignore_index=True)

            # Filter predictions for the current repeat and fold
            df_pred_ = df_pred.loc[(df_pred['fold'] == fold) & (df_pred['repeat'] == repeat), :].drop(
                columns=['y', 'fold', 'repeat', 'y_pred'])

            # Calculate correlations of sensor features with EMR features and store the highest
            for feature in top_10_sensor_stats_features.index:

                features = emr_features + [feature]
                corr_matrix = df_pred_[features].corr(method='pearson')
                feature_corr = corr_matrix[feature].drop(feature)

                # Find the highest absolute correlation and store
                highest_corr = feature_corr.abs().max()
                correlation_df[feature] = highest_corr

            # Concatenate correlation data
            cor_df_stats = pd.concat([cor_df_stats, correlation_df], ignore_index=True)

            # Calculate correlations for neural network features if they exist
            if check_nn:

                # Initialize a DataFrame to store correlation coefficients for the current iteration
                correlation_df = pd.DataFrame(index=[0])

                # calculate correlations of sensor features with emr features and store highest
                for index, feature in enumerate(sorted_nn_features[:10]):

                    features = emr_features + [feature]
                    corr_matrix = df_pred_[features].corr(method='pearson')
                    feature_corr = corr_matrix[feature].drop(feature)

                    # Find the highest absolute correlation and store
                    highest_corr = feature_corr.abs().max()
                    correlation_df[f'nn_feature_{index + 1}'] = highest_corr

                # Concatenate correlation data
                cor_df_neural = pd.concat([cor_df_neural, correlation_df], ignore_index=True)

    # Calculate statistics for feature importances
    df_transposed = cor_df_stats.T
    df_stats_summary = pd.DataFrame({
        'stats_Q1': df_transposed.quantile(0.25, axis=1),
        'stats_median': df_transposed.median(axis=1),
        'stats_Q3': df_transposed.quantile(0.75, axis=1)
    }).iloc[::-1].reset_index().rename(columns={'index': 'Feature'})

    # Calculate statistics for feature correlations
    df_transposed = cor_df_neural.T
    df_neural_summary = pd.DataFrame({
        'neural_Q1': df_transposed.quantile(0.25, axis=1),
        'neural_median': df_transposed.median(axis=1),
        'neural_Q3': df_transposed.quantile(0.75, axis=1)
    }).iloc[::-1].reset_index().rename(columns={'index': 'Feature'})
    df_neural_summary['Feature'] = df_neural_summary['Feature'].str.replace(r'^nn_feature_(\d+)', r'Neural feature \1', regex=True)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 12), sharey=False)

    # Plot for stats features
    sns.barplot(
        data=df_stats_summary,
        y='Feature',
        x='stats_median',
        ax=axes[0],
        color='skyblue',
        errorbar=None,
        orient='h'
    )
    axes[0].set_xlabel('Median Value', fontsize=16)
    axes[0].set_ylabel('Feature', fontsize=16)

    # Add error bars for stats features
    for i, row in df_stats_summary.iterrows():
        axes[0].errorbar(
            x=row['stats_median'],
            y=i,  # Position of the bar
            xerr=[[row['stats_median'] - row['stats_Q1']], [row['stats_Q3'] - row['stats_median']]],
            fmt='none',
            color='black',
            capsize=5
        )

    # Plot for neural features
    sns.barplot(
        data=df_neural_summary,
        y='Feature',
        x='neural_median',
        ax=axes[1],
        color='lightgreen',
        errorbar=None,
        orient='h'
    )
    axes[1].set_xlabel('Median Value', fontsize=16)
    axes[1].set_ylabel('')  # No ylabel for the second plot

    # Add error bars for neural features
    for i, row in df_neural_summary.iterrows():
        axes[1].errorbar(
            x=row['neural_median'],
            y=i,  # Position of the bar
            xerr=[[row['neural_median'] - row['neural_Q1']], [row['neural_Q3'] - row['neural_median']]],
            fmt='none',
            color='black',
            capsize=5
        )

    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(-0.5, 9.5)
    axes[0].set_xticks([i * 0.1 for i in range(11)])  # Generates ticks from 0.0 to 1.0 at 0.1 intervals
    axes[0].tick_params(axis='y', labelsize=16)
    axes[0].tick_params(axis='x', labelsize=16)
    axes[0].grid(axis='x', color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
    axes[0].set_xlabel('Max absolute correlation coefficient', fontsize=16)
    axes[0].set_ylabel(None)
    axes[0].text(-0.40, 1.02, "a)", transform=axes[0].transAxes, fontsize=20, fontweight='bold', va='top',
                 ha='right')

    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(-0.5, 9.5)
    axes[1].set_xticks([i * 0.1 for i in range(11)])  # Generates ticks from 0.0 to 1.0 at 0.1 intervals
    axes[1].tick_params(axis='x', labelsize=16)
    axes[1].tick_params(axis='y', labelsize=16)
    axes[1].grid(axis='x', color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
    axes[1].set_xlabel('Max absolute correlation coefficient', fontsize=16)
    axes[1].set_ylabel(None)
    axes[1].text(-0.05, 1.02, "b)", transform=axes[1].transAxes, fontsize=20, fontweight='bold', va='top',
                 ha='right')

    plt.tight_layout()
    plt.show()

    return df_stats_summary, df_neural_summary
