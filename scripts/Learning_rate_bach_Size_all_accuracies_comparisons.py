# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 16:29:09 2025

@author: H.A.R
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read CSV
data = pd.read_csv(r'E:\GRR\LR_BS_Results\Accuracies_based_on_complete_data.csv')

# Parameters
learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [32, 64, 128, 256]
methods = ['Novel GRR', 'AdaptiveL1L2', 'FixedL1', 'FixedL2', 'ElasticNet', 'MLP']
colors = ['b', 'g', 'r', 'c', 'm', 'y']
line_styles = [':', '--', '-.', ':', '--', '-.']
markers = ['o', 's', '^', 'D', 'v', '*']

# Metrics to plot
accuracy_metrics = [
    ('Average_Test_Accuracy', 'Test Accuracy'),
    ('Average_Validation_Accuracy', 'Validation Accuracy'),
    ('Average_Training_Accuracy', 'Training Accuracy')
]

# Create subplots (3 rows × 3 columns) with adjusted figure size
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 16), sharey=True)
plt.subplots_adjust(hspace=0.3, wspace=0.2)  # Adjusted spacing between subplots
label_font = {'fontsize': 14}
legend_font = {'title_fontsize': 14, 'fontsize': 12}
titles = ['Learning Rate: 0.01', 'Learning Rate: 0.001', 'Learning Rate: 0.0001']

# Adjust global plot parameters
plt.rcParams['axes.titlepad'] = 15  # More space above subplot titles
plt.rcParams['lines.markersize'] = 8  # Slightly smaller markers
plt.rcParams['pdf.fonttype'] = 42  # Best quality text for PDF
plt.rcParams['ps.fonttype'] = 42

for row_idx, (metric_col, ylabel) in enumerate(accuracy_metrics):
    for col_idx, lr in enumerate(learning_rates):
        ax = axes[row_idx][col_idx]
        lr_data = data[data['Learning_Rate'] == lr]

        for method, color, ls, marker in zip(methods, colors, line_styles, markers):
            method_data = lr_data[lr_data['Models'] == method].sort_values('Batch_Size')
            if not method_data.empty:
                jitter = np.random.uniform(-0.0001, 0.0001, len(method_data))
                ax.plot(
                    method_data['Batch_Size'],
                    method_data[metric_col] + jitter,
                    linestyle=ls,
                    color=color,
                    marker=marker,
                    label=method,
                    alpha=0.7,
                    linewidth=2
                )

        ax.set_title(titles[col_idx], fontsize=14, pad=15)  # Added pad for title spacing
        ax.set_xlabel('Batch Size', fontsize=12)
        if col_idx == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_xticks(batch_sizes)
        ax.set_ylim(0.99, 1.001)
        ax.grid(True, which="both", ls="--", alpha=0.2)

        if row_idx == 0 and col_idx == 2:
            ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left', **legend_font)

# Adjust main title position and padding
plt.tight_layout(rect=[0, 0, 1, 0.96])  # More space for suptitle
fig.suptitle('Accuracy (Test, Validation, Training) vs Batch Size at Different Learning Rates - Based on Complete Data', 
             fontsize=16, y=0.98)  # y-position adjusted

# Create output directory if it doesn't exist
output_dir = r'E:\GRR\LR_BS_Results\visuals'
os.makedirs(output_dir, exist_ok=True)

# Save both formats
base_filename = os.path.join(output_dir, 'all_accuracies')

# Save as PDF (vector format)
pdf_path = f"{base_filename}.pdf"
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
print(f"✅ PDF plot saved at: {pdf_path}")

# Save as PNG (raster format) - removed optimize parameter
png_path = f"{base_filename}.png"
plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
print(f"✅ PNG plot saved at: {png_path}")

plt.show()