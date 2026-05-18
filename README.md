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
- fgsea - for ranked enrichment analysis
- GO.db - For extracting data from GO database 
- gbmt - For group-based trajectory modeling

### Python enviroments:
- joint_rpca.yml
- ml_environment.yml

## Repository Structure
~~~text
├── README.md                                  # Prerequisites and repository structure
├── env                                        # Directory for enviroment files
│   ├── joint_rpca.yml                         # Environment to for generating/plotting Joint-RPCA results
│   └── ml_environment.yml                     # Environment to for generating/plotting XGBoost results
├── Figure_generation                          # Directory for code used for figure generation
│   ├── Figure2.Rmd                            # Code to generate Figure 2A-D
│   ├── Figure3.ipynb                          # Code to generate Figure 3A-B (env: joint_rpca.yml)
│   ├── Figure4_pt1.ipynb                      # Code to generate Figure 4A (env: ml_environment.yml)
│   ├── Figure4_pt2.Rmd                        # Code to generate Figure 4BC
│   ├── Figure5.Rmd                            # Code to generate Figure 5A-C
│   ├── Figure6_pt1.Rmd                        # Code to generate Figure 6A-D and F-G
│   ├── Figure6_pt2.ipynb                      # Code to generate Figure 6E (env: joint_rpca.yml)
│   ├── FigureS1_pt1.ipynb                     # Code to generate Figure S1A (env: joint_rpca.yml)
│   ├── FigureS1_pt2.Rmd                       # Code to generate Figure S1B
│   ├── FigureS2.ipynb                         # Code to generate Figure S2 (env: joint_rpca.yml)
│   ├── FigureS3.ipynb                         # Code to generate Figure S3  (env: joint_rpca.yml)
│   ├── FigureS4.ipynb                         # Code to generate Figure S4  (env: ml_environment.yml)
│   ├── FigureS5.Rmd                           # Code to generate Figure S5
│   ├── FigureS6.Rmd                           # Code to generate Figure S6
│   ├── FigureS7.Rmd                           # Code to generate Figure S7
│   ├── FigureS8.Rmd                           # Code to generate Figure S8
│   ├── FigureS9.Rmd                           # Code to generate Figure S9
│   ├── FigureS10_pt.Rmd                       # Code to generate Figure S10AB
│   ├── FigureS10_pt2.ipynb                    # Code to generate Figure S10C
│   ├── FigureS11.Rmd                          # Code to generate Figure S11
│   ├── FigureS12.Rmd                          # Code to generate Figure S12
│   ├── FigureS13.Rmd                          # Code to generate Figure S14 (env: ml_environment.yml)
│   ├── FigureS14.ipynb                        # Code to generate Figure S2 (env: joint_rpca.yml)
│   ├── plotting_setup.R                       # Script sourced to load data for R figure generation
│   ├── plotting_functions.R                   # Script sourced to load functions for R figure generation
│   ├── joint_rpca_output_preprocessing.py     # Script sourced to load data for figures utilizing Joint-RPCA data 
│   ├── joint_rpca_functions.py                # Script sourced to load functions for figures utilizing Joint-RPCA data
│   └── ml_plotting_functions.py               # Script sourced to load functions for figures utilizing XGBOOST data
├── Trajectory_analysis                        # Directory for code used for group-based trajectory modeling analysis
│   ├── Misame_growth_trajectory_analysis.Rmd  # Code for group-based trajectory analysis for MISAME
│   └── Vital_growth_trajectory_analysis.Rmd   # Code for group-based trajectory analysis for Mumta-LW (aka VITAL)
├── Joint_RPCA                                 # Directory for code used for RPCA analysis (env: joint_rpca.yml for all files)
│   ├── 1.0-misame-data-match.ipynb            # Code to preprocess data for MISAME  (env: ml_environment.yml)
│   ├── 1.1-misame-joint-rpca.ipynb            # Code to run joint-RPCA for MISAME  (env: ml_environment.yml)
│   ├── 2.0-vital-data-match.ipynb             # Code to preprocess data for Mumta-LW (aka VITAL; env: ml_environment.yml)
│   ├── 2.1-vital-joint-rpca.ipynb             # Code to run joint-RPCA for Mumta-LW (aka VITAL; env: ml_environment.yml)
│   └── functions.py                           # Functions to assist in running Joint-RPCA analysis
└── Supervised_machine_learning                # Directory for code used for supervised machine learning analysis
    ├── machine_learning_analysis.ipynb        # Code used to train machine learning models  (env: ml_environment.yml)
    ├── ml_functions.py                        # Functions used for machine learning analysis
    └── load_untargeted_data.py                # Script sourced to load data supervised machine learning analysis




