# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 22:22:41 2025

@author: H.A.R
"""

import pandas as pd

# File paths
common_sequences_file = "E:/CDS/Best_Hits/Common_Sequences.txt"
species_file = "E:/CDS/Best_Hits/Complete_data_Set_4_Species.csv"
output_file = "E:/CDS/Best_Hits/RBH_Filtered_Data_Set_4_Species.csv"

# Load common sequences from the text file
common_sequences = set()
with open(common_sequences_file, 'r') as f:
    for line in f:
        common_sequences.update(line.strip().split("\t"))  # Store all sequence IDs in a set

print(f"✅ Loaded {len(common_sequences)} unique sequence IDs from {common_sequences_file}")

# Load the 4_Species CSV file
df_species = pd.read_csv(species_file)

# Ensure first column name is correct
if 'Sequence_ID' not in df_species.columns:
    raise ValueError("❌ Column 'Sequence_ID' not found in the 4_Species.csv file.")

# Filter rows where Sequence_ID is in common_sequences
filtered_df = df_species[df_species['Sequence_ID'].isin(common_sequences)]

# Save filtered data to a new CSV file
filtered_df.to_csv(output_file, index=False)

# Print summary
print(f"✅ Extracted {filtered_df.shape[0]} matching rows.")
print(f"✅ Results saved in: {output_file}")
