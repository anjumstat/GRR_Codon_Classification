# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:18:06 2025

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 2025
Updated by: ChatGPT for H.A.R
"""

import os
from Bio import SeqIO

# Input and output directory paths
input_dir = r"E:\Bras"
output_dir = os.path.join(input_dir, "Filtered")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each FASTA file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".fa") or filename.endswith(".fasta"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename.rsplit('.', 1)[0] + "_filtered.fasta")

        with open(output_file, "w") as out_fasta:
            for record in SeqIO.parse(input_file, "fasta"):
                if len(record.seq) % 3 == 0:
                    SeqIO.write(record, out_fasta, "fasta")

        print(f"Processed: {filename} -> Saved: {output_file}")

print("All filtering operations complete.")
