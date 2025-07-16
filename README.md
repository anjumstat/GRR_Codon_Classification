# GRR_Codon_Classification
Gradient-Responsive Regularization A Deep Learning Framework for Codon Frequency Based Classification of Evolutionarily Conserved Genes

# Data Preprocessing Pipeline
Step-by-Step Filtration Scripts
1. Clean_1.py - Frame Validation

Filters DNA sequences to retain only those with lengths divisible by 3 (valid codon frames)

Output: filtered_length.fasta

2. Clean_2.py - Base Validation

Removes sequences containing non-standard nucleotides (non-ACGT characters)

Output: filtered_bases.fasta

3. Clean_3.py - Start Codon Check

Validates presence of "ATG" start codon at sequence beginnings

Output: filtered_start.fasta

4. Clean_4.py - Stop Codon Verification

Checks for valid stop codons (TAA/TAG/TGA) at sequence ends

Output: filtered_stop.fasta

5. Clean_5.py - Comprehensive QC

Final filtration removing sequences with:

Premature stop codons

Non-standard amino acids

DNA/amino acid length mismatches

Output: final_clean_sequences.fasta
# Reciprocal Best Hits (RBH) Identification Pipeline
Overview
This pipeline identifies reciprocal best hits between two genomes using BLASTn and Python post-processing. The workflow:

Creates BLAST databases for both species

Performs bidirectional BLAST searches

Identifies reciprocal best matches

Quick Start
# 1. Create BLAST databases
makeblastdb -in SpeciesA.fasta -dbtype nucl -out SpeciesA_db
makeblastdb -in SpeciesB.fasta -dbtype nucl -out SpeciesB_db

# 2. Run bidirectional BLAST
blastn -query SpeciesA.fasta -db SpeciesB_db -out SpeciesA_vs_SpeciesB.out -outfmt 6 -max_target_seqs 1
blastn -query SpeciesB.fasta -db SpeciesA_db -out SpeciesB_vs_SpeciesA.out -outfmt 6 -max_target_seqs 1

# 3. Find RBH pairs
python scripts/find_rbh.py \
       -a SpeciesA_vs_SpeciesB.out \
       -b SpeciesB_vs_SpeciesA.out \
       -o RBH_SpeciesA-SpeciesB.csv

File Descriptions
File	Purpose
*.fasta	Input genome sequences
*_db	BLAST databases
*_vs_*.out	Raw BLAST results (tab-separated)
RBH_*.csv	Final reciprocal best hits
# Pipeline Steps
# 1. Database Preparation
makeblastdb -in [GENOME].fasta -dbtype nucl -out [PREFIX]_db
blastn -query [QUERY].fasta -db [TARGET]_db -out [QUERY]_vs_[TARGET].out \
       -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore" \
       -max_target_seqs 1
# Input: Two BLAST result files
# Output: CSV with reciprocal pairs meeting:
#   - Each sequence is the other's top hit
#   - Preserves all BLAST alignment metrics
# Output Columns
The final RBH CSV contains all original BLAST columns plus:

Query_ID: Sequence from SpeciesA

Subject_ID: Matching sequence from SpeciesB

Alignment metrics (Identity%, E-value, Bit score, etc.)
# Example Structure
project/
├── genomes/
│   ├── Oryza_sativa.fasta
│   └── Triticum.fasta
├── blast_results/
│   ├── Oryza_vs_Triticum.out
│   └── Triticum_vs_Oryza.out
└── rbh_results/
    └── Oryza-Triticum_RBH.csv
# Dependencies
BLASTn 2.16.0
python: 3.12.6

