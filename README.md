# FMT-16S-Analysis-2026
Code for the analysis and figure generation (Figures 1 and 2) for the 16S microbiome analysis performed in the manuscript: "Fecal microbiota transplantation from healthy donors reverses antibiotic-induced hematopoietic, barrier, and colonization deficits in mice"

Primary 16S analysis and figure generation was performed in R in the notebook FMT_16S_analysis.rmd, using the following R packages: dplyr, phyloseq, readxl, stringr, vegan, ggplot2, ggpubr, ANCOMBC, ggprism. Notebook output can be found in the FMT_16S_analysis.html file. In this workflow, 16S data is processed, alpha and beta diversity and ANCOMBC2 analysis is performed, and data is formatted for RPCA analysis (see RPCA.ipynb). Also contains code used to generate Figure 1 and 2, as well as supplementary S1. All other plots in this manuscript were generated in PRISM.    

RPCA analysis was performed in the RPCA.ipynb notebook in the RPCA_3 folder, which was used to execute bash commands for biom-format(v. 2.1.15), and qiime2 (v. 2024.5.1), which was used as an interface for gemelli (v. 0.0.12). 
