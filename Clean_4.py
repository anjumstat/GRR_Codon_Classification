# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:28:07 2025

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 2025
Script to remove sequences with non-standard DNA bases (only A, T, C, G allowed)
"""

import os
from Bio import SeqIO

# Input and output directory paths
input_dir = r"E:\Bras\Filtered"
output_dir = r"E:\Bras\Cleaned"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Allowed standard DNA bases
valid_bases = set("ATCGatcg")

# Process each FASTA file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".fa") or filename.endswith(".fasta"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename.rsplit('.', 1)[0] + "_cleaned.fasta")

        removed_count = 0
        kept_records = []

        for record in SeqIO.parse(input_file, "fasta"):
            if set(record.seq).issubset(valid_bases):
                kept_records.append(record)
            else:
                removed_count += 1

        # Write only the valid records
        with open(output_file, "w") as out_fasta:
            SeqIO.write(kept_records, out_fasta, "fasta")

        print(f"Processed: {filename}")
        print(f"  → Sequences removed (non-standard bases): {removed_count}")
        print(f"  → Valid sequences saved to: {output_file}")

print("Cleaning complete for all files.")
