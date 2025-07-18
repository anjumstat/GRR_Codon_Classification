# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 21:59:42 2025

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 11:37:42 2025
Updated on Jul 13 to preserve full metric names in results

@author: H.A.R
"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene, f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import combinations

# Load dataset
df = pd.read_csv("E:/GRR/Combined_CSVs/Fold_Metrics_Detailed_RBH.csv")

# Metrics to compare (include all relevant sets)
metric_columns = [
    'Train_Accuracy', 'Train_Precision', 'Train_Recall', 'Train_F1', 'Train_MCC',
    'Val_Accuracy', 'Val_Precision', 'Val_Recall', 'Val_F1', 'Val_MCC',
    'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_MCC'
]

# All parameter combinations
param_combinations = df[['Learning_Rate', 'batch_size']].drop_duplicates()

anova_results = []

for _, params in param_combinations.iterrows():
    lr = params['Learning_Rate']
    bs = params['batch_size']

    subset = df[(df['Learning_Rate'] == lr) & (df['batch_size'] == bs)]

    for metric in metric_columns:
        data = subset[['adaptive_method', metric]].dropna()
        if data['adaptive_method'].nunique() < 2:
            continue

        # Test normality per group
        normal = True
        for method in data['adaptive_method'].unique():
            vals = data[data['adaptive_method'] == method][metric]
            if len(vals) < 3 or shapiro(vals)[1] < 0.05:
                normal = False
                break

        # Test equal variances
        groups = [group[metric].values for name, group in data.groupby('adaptive_method')]
        if len(groups) < 2:
            continue
        levene_p = levene(*groups).pvalue

        if normal and levene_p > 0.05:
            # Parametric ANOVA
            model = ols(f'{metric} ~ C(adaptive_method)', data=data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            p_val = anova_table['PR(>F)'][0]
            test_used = "ANOVA"
        else:
            # Non-parametric Kruskal-Wallis test
            test_stat, p_val = kruskal(*groups)
            test_used = "Kruskal-Wallis"

        # Determine best method (highest mean)
        mean_scores = data.groupby('adaptive_method')[metric].mean().sort_values(ascending=False)
        best_method = mean_scores.index[0]

        # Append result with full metric name
        result = {
            'Learning_Rate': lr,
            'Batch_Size': bs,
            'Metric': metric,  # Preserve full column name
            'Test_Used': test_used,
            'p_value': round(p_val, 5),
            'Best_Method': best_method,
            'Best_Mean': round(mean_scores.iloc[0], 4),
            'Significant': p_val < 0.05
        }

        # Add second best info if exists
        if len(mean_scores) > 1:
            result['Second_Best'] = mean_scores.index[1]
            result['Mean_Difference'] = round(mean_scores.iloc[0] - mean_scores.iloc[1], 4)
        else:
            result['Second_Best'] = None
            result['Mean_Difference'] = None

        anova_results.append(result)

# Convert to DataFrame
anova_df = pd.DataFrame(anova_results)

# Save to CSV
anova_df.to_csv("E:/GRR/post_hoc/RBH_Statistical_Comparison_ANOVA_Kruskal.csv", index=False)

print(f"✅ Multi-group comparison results saved. Total comparisons: {len(anova_df)}")
print(anova_df.head())
