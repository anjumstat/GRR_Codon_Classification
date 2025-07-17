# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 22:13:19 2025

@author: H.A.R
"""

import networkx as nx
import os

# Directory where RBH files are stored
rbh_dir = "E:/CDS/Best_Hits"

# Output file for common sequences
output_file = "E:/CDS/Best_Hits/Common_Sequences.txt"

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Initialize an undirected graph
G = nx.Graph()

# Load all RBH files (e.g., RBH_Brach_Hordeum.txt)
rbh_files = [f for f in os.listdir(rbh_dir) if f.startswith("RBH_") and f.endswith(".txt")]

print(f"✅ Found {len(rbh_files)} RBH files.")

# Add edges from all RBH files
for rbh_file in rbh_files:
    file_path = os.path.join(rbh_dir, rbh_file)
    with open(file_path, 'r') as f:
        for line in f:
            seq1, seq2 = line.strip().split()  # Ensure each line contains exactly two sequence IDs
            G.add_edge(seq1, seq2)

# Find clusters of similar sequences
common_sequences = []
for component in nx.connected_components(G):
    species_in_cluster = set()
    for seq in component:
        species = seq.split("_")[0]  # Extract species identifier from sequence ID
        species_in_cluster.add(species)

    if len(species_in_cluster) == 4:  # Ensure all four species are represented
        common_sequences.append(component)

# Save common sequences to file
with open(output_file, 'w') as f:
    for cluster in common_sequences:
        f.write("\t".join(cluster) + "\n")

# Print summary
print(f"✅ {len(common_sequences)} common sequence clusters found across all four species.")
print(f"✅ Results saved in: {output_file}")
