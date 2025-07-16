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

# 3. RBH Identification 
Input Requirements
Two BLAST output files (format -outfmt 6) from reciprocal searches:

SpeciesA_vs_SpeciesB.out (Query: SpeciesA, Target: SpeciesB)

SpeciesB_vs_SpeciesA.out (Query: SpeciesB, Target: SpeciesA)
Execution
python scripts/RBH_2_Species.py
Automatically processes files in input_folder and saves results to output_folder
What It Does
Reads and Standardizes BLAST results:

Parses tab-separated -outfmt 6 outputs

Trims whitespace from gene IDs

Handles column naming consistency

Identifies Reciprocal Best Hits:

Validates that Sequence A's top hit is Sequence B and vice versa

Preserves all alignment metrics:

Percent identity

Alignment length

E-value

Bit score

Coverage positions

Outputs Results:

Generates similar_sequences.csv with matched pairs

Includes all original BLAST columns

Adds suffixes (_A2B, _B2A) to distinguish reciprocal directions
Customization
To adapt for different file paths:

Modify these variables in the script:
input_folder = "path/to/blast_results"  # Line 7
output_folder = "path/to/rbh_output"   # Line 8
