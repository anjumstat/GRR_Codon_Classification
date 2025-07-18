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
# Calculate_codon_frequencies.py
Codon Frequency Calculator
This script calculates absolute codon frequencies for coding sequences (CDS) in FASTA files across four species:

Triticum aestivum (Wheat)

Oryza sativa (Rice)

Hordeum vulgare (Barley)

Brachypodium distachyon

Features
Processes multiple FASTA files in batch mode

Outputs CSV files with sequence IDs and absolute counts for all 64 codons

Maintains consistent codon order (lexicographic) for easy comparison

Input
FASTA files (.fa or .fasta extension) containing CDS sequences

Files should be placed in the specified input directory (E:\Bras\3_cond)

Output
For each input file:

A CSV file named [filename]_codon_frequencies.csv containing:

Sequence_ID: Original FASTA header

Columns for all 64 codons (e.g., AAA, AAC, ..., TTT) with absolute counts

Usage
Place FASTA files in the input directory

Run the script:

bash
python codon_frequency_calculator.py
Output CSVs will be saved in [input_dir]/codon_frequencies/

Dependencies
Python 3

Biopython (pip install biopython)

pandas (pip install pandas)

Example Output
Sequence_ID Species	             AAA	AAC	...	TTT
Gene_1	    Triticum_aestivum    12	5	...	8
Gene_2	    Triticum_aestivum    7	3	...	2

Note: This code will be run for four fasta input files, which will provide four .csv files for four species (crops). After this  accumlate these files in one file and add a column Label by Labeling 1 for Triticum aestivum, 2 for oryza_sativa, 3 for Hordium Volgare, 4 for Bracypodium Distachyon. This is a complete data set for four crops. 
# RBH_Data_Extractor.py
This script extracts and filters gene data for reciprocal best hit (RBH) orthologs from a complete dataset, creating a specialized subset for ortholog analysis.

Input Files
Common_Sequences.txt

Contains tab-separated sequence IDs of RBH orthologs across all four species

Generated from previous ortholog cluster identification step

Complete_data_Set_4_Species.csv

Comprehensive dataset containing all gene information

Must include:

Sequence_ID column (matching IDs in Common_Sequences.txt)

Codon frequency data

Species annotations

Output File
RBH_Filtered_Data_Set_4_Species.csv:

Contains only genes identified as RBH orthologs

Preserves all columns from the complete dataset

Ready for downstream comparative analysis

Usage
bash
python filter_rbh_data.py
Dependencies
Python 3

pandas (pip install pandas)

Expected Output
# Final data
Now we have two .csv files one contains complete data for all genes across four species named Complete_data_Set_4_Species.csv, which have a total number of 253, 076 rows. Second .csv file named RBH_Filtered_Data_Set_4_Species.csv, which contains 25,152 orthologous genes data across four species. Now these two data sets will be proceed for model fitting. 
# GRR_MLP_Variants_Analysis.py - The code to produce final Comparative Analysis of Regularization Methods for Species Classification based on Complete Data set and RBH data set. 
Note: Delete first two columns and name third column as "Species" of .csv input files before running the code. Also due to limitations from github I have splitted Complete Data set file in two parts, first part named Complete_data_Set_4_Species_part1.csv and second part named Complete_data_Set_4_Species_part1.csv. Combine these files before running code on complete data set.   

This script performs a comprehensive comparison of multiple regularization approaches, including a novel Gradient-Responsive Regularizer (GRR), for classifying plant species based on codon usage patterns.

Key Features
Novel GRR Implementation: Custom regularizer that dynamically adjusts L1/L2 penalties based on gradient behavior

Six Regularization Methods Compared:

Novel GRR (Our proposed method)

Adaptive L1L2

Fixed L1

Fixed L2

ElasticNet

Standard MLP

Hyperparameter Optimization: Evaluates multiple learning rates and batch sizes

Robust Evaluation: 10-fold cross validation with comprehensive metrics

Input Requirements
Complete_data_Set_4_Species.csv containing:

Sequence IDs

Codon frequency data (64 features)

Species labels (1-4)

Output Structure
text
MLP_RBH_all3/
├── lr_[rate]_bs_[size]/
│   ├── results_Novel/
│   ├── results_Adaptive/
│   ├── ... (other methods)
│   │   ├── Best_Model_Metrics.csv
│   │   ├── Average_Metrics.csv
│   │   ├── Fold_Metrics_Detailed.csv
│   │   ├── Training_Time.csv
│   │   ├── Confusion_Matrix.png
│   │   └── Best_*_Model.h5
└── (analysis files)
Key Metrics Reported
For each method and hyperparameter combination:

Accuracy, Precision, Recall, F1, MCC

Training time

Validation curves

Usage
Configure paths in the script:

python
DATA_PATH = 'E:/GRR/Complete_data_Set_4_Species.csv'
BASE_DIR = 'E:/GRR/MLP_RBH_all3'
Adjust hyperparameter search space if needed:

python
LEARNING_RATES = [0.01, 0.001, 0.0001]
BATCH_SIZES = [32, 64, 128, 256]
Run the script:

bash
python species_classification_analysis.py
Dependencies
Python 3.7+

TensorFlow 2.x

scikit-learn

pandas

NumPy

Matplotlib

Seaborn

SciPy

Novel GRR Implementation Details
The Gradient-Responsive Regularizer features:

Dynamic adaptation based on gradient variance

Feature-specific modulation

Stability-aware penalty adjustment

Historical tracking of regularization parameters
# Learning_rate_bach_Size_all_accuracies_comparisons.py
To generate Figures 6 and 7 of manuscript, please execute the script Learning_rate_batch_Size_all_accuracies_comparisons.py.

The script requires the following input files:

Accuracies_based_on_complete_data.csv - containing model accuracy results from the complete dataset

Accuracies_based_on_RBH_data.csv - containing model accuracy results from the RBH (Reciprocal Best Hits) dataset

Note: These CSV files aggregate the model performance metrics across all combinations of batch sizes and learning rates from the model fitting process.
# Plot_Training_Validation_Accuracy_Loss.py
The script Plot_Training_Validation_Accuracy_Loss.py generates Figure 8 and Figure 9 in the manuscript. To execute this script, provide the path to the output folder containing the training and validation results. These results are produced by running GRR_MLP_Variants_Analysis.py for a specific combination of batch size and learning rate.
# Over_fitting_Plots.py
To produce Figure 10 and Figure 11 in the manuscript, execute the script Over_fitting_Plots.py. This script analyzes overfitting trends by aggregating results across all combinations of batch sizes and learning rates from the model fitting process.

Input Requirements:
The script requires the following data files, which consolidate training and validation results:

Complete_Train_Valid_Overfitting.csv – Results from the complete dataset

RBH_Train_Valid_Overfitting.csv – Results from the RBH (Reciprocal Best Hits) dataset

These files are generated by compiling outputs from GRR_MLP_Variants_Analysis.py, where each run corresponds to a specific batch size and learning rate combination.

# Kruskal_Walis_ANOVA_Test.py
Statistical Testing Implementation
The script Kruskal_Wallis_ANOVA_Test.py performs nonparametric (Kruskal-Wallis) or parametric (ANOVA) statistical testing, automatically selecting the appropriate test based on data distribution assumptions.
Input Data Requirements:
The analysis requires two consolidated metric files:
Fold_Metrics_Detailed_comp.csv - Complete dataset metrics
Fold_Metrics_Detailed_RBH.csv - RBH-filtered dataset metrics
These CSV files aggregate cross-validation results from model fitting runs executed via GRR_MLP_Variants_Analysis.py, by combining outputs across all experimental conditions for both complete and RBH-filtered datasets.


