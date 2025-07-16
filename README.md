# GRR_Codon_Classification
Gradient-Responsive Regularization A Deep Learning Framework for Codon Frequency Based Classification of Evolutionarily Conserved Genes

Data Preprocessing Pipeline
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
