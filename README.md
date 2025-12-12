# Code to generate figures for: Universal Patterns of Human Milk Composition Link Mammary Gland Function to Infant Growth

## Prerequisites
### R packages:
- tidyverse - For dataframe manipulation
- ggplot2 - For plotting
- ggpubr - For publication-ready plotting
- readxl - for reading excel documents
- Hmisc - for calculating spearman R matricies with p values
- ComplexHeatmap - for building heatmaps with complex text labels
- circlize - dependency for ComplexHeatmap
- ggplotify - for saving heatmaps as ggplot objects for ggarrange
- epitools - For risk ratio analysis
- zscorer - for hcaz calculations
- purrr - for complex iteration
### Python enviroments:
- joint_rpca.yml
- ml_environment.yml

## Repository Structure
~~~text
├── README.md                          # Prerequisites and repository structure
├── env                                # Directory for enviroment files
│   ├── joint_rpca.yml                 # Environment to for generating/plotting Joint-RPCA results
│   ├── ml_environment.yml             # Environment to for generating/plotting XGBoost results
├── Figure2.Rmd                        # Code to generate Figure 2A-D
├── Figure3.ipynb                      # Code to generate Figure 3A-B (env: joint_rpca.yml)
├── Figure4_pt1.ipynb                  # Code to generate Figure 4A (env: ml_environment.yml)
├── Figure4_pt2.Rmd                    # Code to generate Figure 4C
├── Figure5.Rmd                        # Code to generate Figure 5A-C
├── Figure6.Rmd                        # Code to generate Figure 6A-D
├── FigureS1_pt1.ipynb                 # Code to generate Figure S1A (env: joint_rpca.yml)
├── FigureS1_pt2.Rmd                   # Code to generate Figure S1B
├── FigureS2.ipynb                     # Code to generate Figure S2 (env: joint_rpca.yml)
├── FigureS3.ipynb                     # Code to generate Figure S3  (env: ml_environment.yml)
├── FigureS4.Rmd                       # Code to generate Figure S4 
├── FigureS5_pt1.Rmd                   # Code to generate Figure S5A and S5C
├── FigureS5_pt2.ipynb                 # Code to generate Figure S5B (env: joint_rpca.yml)
├── FigureS6.Rmd                       # Code to generate Figure S6A-C
├── Figure_S7.ipynb                    # Code to generate Figure S7 (env: ml_environment.yml)
├── plotting_setup.R                   # Script sourced to load data for R figure generation
├── plotting_functions.R               # Script sourced to load functions for R figure generation
├── joint_rpca_output_preprocessing.py # Script sourced to load data for figures utilizing Joint-RPCA data 
├── joint_rpca_functions.py            # Script sourced to load functions for figures utilizing Joint-RPCA data
└── ml_plotting_functions.py           # Script sourced to load functions for figures utilizing XGBOOST data



