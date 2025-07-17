# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 13:37:37 2025

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Compute codon frequencies for all FASTA files in E:\Bras\3_cond
and save the output as CSV files with sequence IDs included.

@author: H.A.R
"""

from Bio import SeqIO
from collections import Counter
import pandas as pd
import os

# Input directory containing the FASTA files
input_dir = r"E:\Bras\3_cond"

# Output directory to save CSVs
output_dir = os.path.join(input_dir, "codon_frequencies")
os.makedirs(output_dir, exist_ok=True)

# Codons in lexicographic order
codon_order = [
    "AAA", "AAC", "AAG", "AAT", "ACA", "ACC", "ACG", "ACT", "AGA", "AGC", "AGG", "AGT",
    "ATA", "ATC", "ATG", "ATT", "CAA", "CAC", "CAG", "CAT", "CCA", "CCC", "CCG", "CCT",
    "CGA", "CGC", "CGG", "CGT", "CTA", "CTC", "CTG", "CTT", "GAA", "GAC", "GAG", "GAT",
    "GCA", "GCC", "GCG", "GCT", "GGA", "GGC", "GGG", "GGT", "GTA", "GTC", "GTG", "GTT",
    "TAA", "TAC", "TAG", "TAT", "TCA", "TCC", "TCG", "TCT", "TGA", "TGC", "TGG", "TGT",
    "TTA", "TTC", "TTG", "TTT"
]

# Loop through each FASTA file
for filename in os.listdir(input_dir):
    if filename.endswith(".fa") or filename.endswith(".fasta"):
        input_file = os.path.join(input_dir, filename)
        data = []

        # Read sequences and compute codon frequencies
        for record in SeqIO.parse(input_file, "fasta"):
            seq = str(record.seq)
            codon_counter = Counter()
            for i in range(0, len(seq), 3):
                codon = seq[i:i+3]
                if len(codon) == 3:
                    codon_counter[codon] += 1
            row = [record.id] + [codon_counter.get(codon, 0) for codon in codon_order]
            data.append(row)

        # Create DataFrame
        df = pd.DataFrame(data, columns=["Sequence_ID"] + codon_order)

        # Output CSV path
        output_csv = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_codon_frequencies.csv")
        df.to_csv(output_csv, index=False)

        print(f"Saved codon frequencies to: {output_csv}")
