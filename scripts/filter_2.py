# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 2025

@author: H.A.R
"""

from Bio import SeqIO
import os

# Input and output directory paths
input_dir = r"E:\Bras\Cleaned"
output_dir = r"E:\Bras\ATG_Start"

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over all FASTA files
for filename in os.listdir(input_dir):
    if filename.endswith(".fa") or filename.endswith(".fasta"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename.rsplit('.', 1)[0] + "_ATGstart.fasta")

        removed_count = 0

        with open(output_file, "w") as out_fasta:
            for record in SeqIO.parse(input_file, "fasta"):
                if record.seq.upper().startswith("ATG"):  # Check if sequence starts with ATG
                    SeqIO.write(record, out_fasta, "fasta")
                else:
                    removed_count += 1

        print("File processed:", filename)
        print("→ Sequences removed (not starting with 'ATG'):", removed_count)
        print("→ Filtered sequences saved in:", output_file)

print("\nFiltering complete for all FASTA files.")
