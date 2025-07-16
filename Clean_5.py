# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 13:07:50 2025

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 2025

@author: H.A.R
"""

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data import CodonTable
import os

# Input and output directory paths
input_dir = r"E:\Bras\END_3"
output_dir = r"E:\Bras\3_cond"

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Standard genetic code
standard_table = CodonTable.unambiguous_dna_by_name["Standard"]
stop_codons = set(standard_table.stop_codons)
valid_bases = set("ATGC")

# Loop through all FASTA files
for filename in os.listdir(input_dir):
    if filename.endswith(".fa") or filename.endswith(".fasta"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename.rsplit('.', 1)[0] + "_cleaned.fasta")

        removed_count = 0

        with open(output_file, "w") as out_fasta:
            for record in SeqIO.parse(input_file, "fasta"):
                seq = str(record.seq).upper()

                # Skip if contains non-standard DNA bases
                if not set(seq).issubset(valid_bases):
                    removed_count += 1
                    continue

                # Skip if not divisible by 3
                if len(seq) % 3 != 0:
                    removed_count += 1
                    continue

                codons = [seq[i:i+3] for i in range(0, len(seq), 3)]

                # Count stop codons in entire sequence
                stop_count = sum(1 for codon in codons if codon in stop_codons)
                if stop_count > 1:
                    removed_count += 1
                    continue

                # Try translation to amino acids
                try:
                    protein = Seq(seq).translate(to_stop=False, table="Standard")
                    # Remove if it translates to nothing or contains '*'
                    if "*" in protein[:-1] or len(protein) == 0:
                        removed_count += 1
                        continue
                except:
                    removed_count += 1
                    continue

                # Passed all filters → write to output
                SeqIO.write(record, out_fasta, "fasta")

        print("File processed:", filename)
        print("→ Sequences removed (non-standard, bad stop codons, or bad translation):", removed_count)
        print("→ Clean sequences saved to:", output_file)

print("\nAdvanced filtering complete for all FASTA files.")
