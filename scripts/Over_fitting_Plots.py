# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 17:27:20 2025
Updated to use Fold-based CSV input format with electric pink color and figure title
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.ticker as ticker

# === Load CSV ===
csv_file = r'E:\GRR\LR_BS_Results\Complete_Train_Valid_Overfitting.csv'
df = pd.read_csv(csv_file)

# Preview data
print("Data preview:\n", df.head())

# Group by method
method_groups = df.groupby('adaptive_method')
method_names = list(method_groups.groups.keys())

# Layout
cols = 2
rows = math.ceil(len(method_names) / cols)

fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
axes = axes.flatten()
plot_idx = 0

# Plot per method
for method_name, group in method_groups:
    folds = group['Fold'].values
    train_acc = group['Training_Accuracy'].values
    val_acc = group['Validation_Accuracy'].values

    if len(train_acc) != len(val_acc):
        print(f"Length mismatch in method {method_name}. Skipping.")
        continue

    acc_gap = train_acc - val_acc

    ax = axes[plot_idx]
    plot_idx += 1
    ax.plot(folds, acc_gap, label='Train - Val Accuracy Gap', color='#FF1493', marker='o', linewidth=2)
    ax.axhline(0, linestyle='--', color='gray', linewidth=1)
    ax.set_title(f"{method_name} - Overfitting Detection", fontsize=12)
    ax.set_xlabel("Fold", fontsize=10)
    ax.set_ylabel("Accuracy Gap", fontsize=10)
    ax.grid(True)
    ax.legend(fontsize=9)

    # Format Y-axis ticks as 0.0000
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

# Hide unused axes
for i in range(plot_idx, len(axes)):
    fig.delaxes(axes[i])

# Add global figure title
fig.suptitle('Overfitting Analysis Across Models - Based on Complete data set', fontsize=16, y=1.02)

plt.tight_layout()
plt.show()
