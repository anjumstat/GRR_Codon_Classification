# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 20:26:43 2025

@author: H.A.R
"""

import pandas as pd
import os

# Define file paths
input_folder = r"E:\CDS\Tri_Oryza"
output_folder = r"E:\CDS\similar"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# File names
file1 = os.path.join(input_folder, "Oryza_vs_Triticum.txt")
file2 = os.path.join(input_folder, "Triticum_vs_Oryza.txt")
output_file = os.path.join(output_folder, "similar_sequences.csv")

# Column names (assuming tab-separated file)
col_names = ["Query_ID", "Subject_ID", "Identity", "Alignment_Length", "Mismatch", "Gap_Open", 
             "Query_Start", "Query_End", "Subject_Start", "Subject_End", "E_Value", "Bit_Score"]

# Read data
df1 = pd.read_csv(file1, sep="\t", names=col_names, engine='python')
df2 = pd.read_csv(file2, sep="\t", names=col_names, engine='python')

# Trim spaces in string columns (if any)
df1["Query_ID"] = df1["Query_ID"].astype(str).str.strip()
df1["Subject_ID"] = df1["Subject_ID"].astype(str).str.strip()
df2["Query_ID"] = df2["Query_ID"].astype(str).str.strip()
df2["Subject_ID"] = df2["Subject_ID"].astype(str).str.strip()

# Swap columns in df2 to match df1 format
df2_renamed = df2.rename(columns={"Query_ID": "Subject_ID", "Subject_ID": "Query_ID"})

# Find common sequences
common = pd.merge(df1, df2_renamed, on=["Query_ID", "Subject_ID"], suffixes=("_Brach_vs_Hordeum", "_Hordeum_vs_Brach"))

# Save to CSV only if there are common sequences
if not common.empty:
    common.to_csv(output_file, index=False)
    print(f"Similar sequences saved to: {output_file}")
else:
    print("No common sequences found.")
