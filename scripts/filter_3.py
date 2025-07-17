# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 13:01:59 2025

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 2025

@author: H.A.R
"""

from Bio import SeqIO
import os

# Input and output directory paths
input_dir = r"E:\Bras\ATG_Start"
output_dir = r"E:\Bras\END_3"

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define valid stop codons
stop_codons = {"TAA", "TAG", "TGA"}

# Loop over all FASTA files
for filename in os.listdir(input_dir):
    if filename.endswith(".fa") or filename.endswith(".fasta"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename.rsplit('.', 1)[0] + "_StopCodonEnd.fasta")

        removed_count = 0

        with open(output_file, "w") as out_fasta:
            for record in SeqIO.parse(input_file, "fasta"):
                seq = str(record.seq).upper()
                if seq[-3:] in stop_codons:  # Check if last 3 bases are stop codon
                    SeqIO.write(record, out_fasta, "fasta")
                else:
                    removed_count += 1

        print("File processed:", filename)
        print("→ Sequences removed (not ending with stop codon):", removed_count)
        print("→ Filtered sequences saved in:", output_file)

print("\nFiltering complete for all FASTA files.")
