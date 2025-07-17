# GRR_Codon_Classification
Gradient-Responsive Regularization A Deep Learning Framework for Codon Frequency Based Classification of Evolutionarily Conserved Genes

# Data Preprocessing Pipeline
Step-by-Step Filtration Scripts
# 1. filter_1.py - Frame Validation

Filters DNA sequences to retain only those with lengths divisible by 3 (valid codon frames)

Output: filtered_length.fasta

# 2. filter_2.py - Base Validation

Removes sequences containing non-standard nucleotides (non-ACGT characters)

Output: filtered_bases.fasta

# 3. filter_3.py - Start Codon Check

Validates presence of "ATG" start codon at sequence beginnings

Output: filtered_start.fasta

# 4. filter_4.py - Stop Codon Verification

Checks for valid stop codons (TAA/TAG/TGA) at sequence ends

Output: filtered_stop.fasta

# 5. filter_5.py - Comprehensive QC

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
# RBH_2_Species.py
This script identifies reciprocal best hits (RBH) from BLASTn results across four species:

Triticum aestivum (Wheat)

Oryza sativa (Rice)

Hordeum vulgare (Barley)

Brachypodium distachyon

RBH pairs are used to infer putative orthologous genes between species.

# Input
Tab-separated BLASTn outputs for all pairwise comparisons (e.g., Triticum_vs_Oryza.txt, Oryza_vs_Triticum.txt, etc.).

Files must include standard BLAST columns:
Query_ID, Subject_ID, Identity, Alignment_Length, E-value, Bit_Score, etc.

# Output
similar_sequences.csv: A consolidated file of reciprocal best hits with alignment statistics.

Usage
Edit paths in the script to point to your input/output folders.

Run the script:

bash
python reciprocal_best_hits.py
Repeat for all species pairs (e.g., Wheat-Barley, Barley-Brachypodium, etc.).

Dependencies
Python 3

pandas (pip install pandas)
# To_Find_common_sequences_across_4_Species
This script identifies conserved orthologous gene clusters across four species using reciprocal best hit (RBH) results from BLASTn. It uses graph theory (NetworkX) to group genes shared among:

Triticum aestivum (Wheat)

Oryza sativa (Rice)

Hordeum vulgare (Barley)

Brachypodium distachyon

Input
RBH files: Text files (e.g., RBH_Brach_Hordeum.txt) generated from the previous RBH analysis (one file per species pair).

Each line must contain two sequence IDs (e.g., Oryza_XYZ123 and Triticum_ABC456).

Output
Common_Sequences.txt: A tab-separated file listing gene clusters where all four species are represented (one cluster per line).

Methodology
Graph Construction:

Genes are nodes, and RBH pairs are undirected edges.

Cluster Detection:

Connected components in the graph represent orthologous groups.

Only clusters containing ≥1 gene from all four species are retained.

Usage
Place all RBH files in the specified directory (E:/CDS/Best_Hits/).

Run the script:

bash
python ortholog_clusters.py
Output will list conserved clusters (e.g., Oryza_G1 Triticum_G2 Hordeum_G3 Brach_G4).

Dependencies
Python 3

networkx (pip install networkx)

Example Output
text
Oryza_XYZ123  Triticum_ABC456  Hordeum_DEF789  Brach_GHI012  
Oryza_XYZ124  Triticum_ABC457  Hordeum_DEF790  Brach_GHI013  
Applications
Comparative genomics studies.

Evolutionary analysis of conserved gene families.

Note: Replace paths/file names as needed. This links directly to your workflow and maintains reproducibility. Let me know if you’d like to add citations or parameter details!



Application
Designed for comparative genomics to identify orthologs across the four species. Adaptable to other datasets by modifying file paths/species names.
