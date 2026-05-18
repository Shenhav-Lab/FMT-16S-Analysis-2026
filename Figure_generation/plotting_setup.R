## CODE FOR CONSISTANT LIBRARY/VARIABLE LOADING FOR FIGURE GENERATION

# Load libraries
library(tidyverse) # For dataframe manipulation
library(ggplot2) # For plotting
library(ggpubr) # For publication-ready plotting
library(readxl) # for reading excel documents
library(Hmisc) # for calculating spearman R matricies with p values
library(ComplexHeatmap) # for building heatmaps with complex text labels
library(circlize) # dependency for ComplexHeatmap
library(ggplotify) # for saving heatmaps as ggplot objects for ggarrange
library(epitools) # For risk ratio analysis
library(zscorer) # for hcaz calculations
library(purrr) # for complex iteration
library(fgsea) # for ranked enrichment analysis
library(GO.db) # For extracting data from GO database 

# Setting paths
IMiC_path = "C://Users/ETTINA03/NYU Langone Health Dropbox/April Jauhal/Shenhav_Lab/"
fig_path = "C://Users/ETTINA03/NYU Langone Health Dropbox/April Jauhal/Shenhav_Lab/IMiC/Figures"

# Targeted data LC/MS + micronutrient data
misame_targeted = read.csv(paste0(IMiC_path, "IMiC/Data/misame_targeted_df_with_ae.csv"), row.names = 1)
vital_targeted = read.csv(paste0(IMiC_path, "IMiC/Data/vital_targeted_df_with_ae.csv"), row.names = 1)
child_targeted = read.csv(paste0(IMiC_path, "IMiC/Data/child_targeted_df_with_ae.csv"), row.names = 1)

# Feature metadata
targeted_spec = read.csv(paste0(IMiC_path,  "IMiC/Data/Targeted_metabolomics/metadata/Milk_Component_Spec_full.csv"), row.names = 1)

# Macronutrient data
misame_macronut.csv = read.csv(paste0(IMiC_path, "IMiC/Code/machine_learning/misame_macronut.csv"),
                                row.names = 1) %>%
  dplyr::rename(SampleID=sampleID)

vital_macronut.csv = read.csv(paste0(IMiC_path, "IMiC/Code/machine_learning/vital_macronut.csv"),
                               row.names = 1) %>%
  dplyr::rename(SampleID=sampleID)
child_macronut.csv = read.csv(paste0(IMiC_path, "IMiC/Code/machine_learning/child_macronut.csv"),
                               row.names = 1) %>%
  tibble::rownames_to_column('SampleID')

# LC/MS data for top untargeted features and feature annotations
misame_104_untargeted_feats = read.csv(paste0(IMiC_path, 'IMiC/Code/machine_learning/misame_untargeted_104.csv'), row.names = 1) %>%dplyr::rename(SampleID=sample_ID)
vital_104_untargeted_feats = read.csv(paste0(IMiC_path, 'IMiC/Code/machine_learning/vital_untargeted_104.csv'), row.names = 1) %>%dplyr::rename(SampleID=sample_ID)
child_104_untargeted_feats = read.csv(paste0(IMiC_path, 'IMiC/Code/machine_learning/child_untargeted_104.csv'), row.names = 1) %>%dplyr::rename(SampleID=sample_ID)

# Loading MSD data 
Mis_MSD = read.csv(file = paste0(IMiC_path, "IMiC/Data/Targeted_metabolomics/MISAME/Bode/MSD_MISAME.csv"), row.names = 1) %>% tibble::rownames_to_column("SampleID")
Vit_MSD = read.csv(file = paste0(IMiC_path, "IMiC/Data/Targeted_metabolomics/Vital/Bode/MSD_VITAL.csv"), row.names = 1) %>% tibble::rownames_to_column("SampleID")
Chd_MSD = read.csv(file = paste0(IMiC_path, "IMiC/Data/Targeted_metabolomics/Child/Bode/MSD_CHILD.csv"), row.names = 1) %>% tibble::rownames_to_column("SampleID") %>% mutate(SampleID=str_remove(SampleID, "-6$"))

# Loading Parsed metadata
misame_full_BM_df = read.delim(file=paste0(IMiC_path, 'IMiC/Data/MISAME/metadata/misame_processed_metadata.tsv'))
vital_full_BM_df = read.delim(file=paste0(IMiC_path, 'IMiC/Data/VITAL/metadata/vital_processed_metadata.tsv'))
child_full_BM_df = read.delim(file=paste0(IMiC_path, 'IMiC/Data/CHILD/metadata/child_processed_metadata.tsv'))

# Loading Supporting metadata generated from other notebooks
misame_supp = read.csv(paste0(IMiC_path, "IMiC/Data/MISAME/metadata/misame_metadata_supp.csv"),
                       row.names = 1) 
misame_supp_margin = read.csv(paste0(IMiC_path, "IMiC/Data/MISAME/metadata/misame_metadata_supp2.csv"),
                              row.names = 1) 
vital_supp = read.csv(paste0(IMiC_path, "IMiC/Data/VITAL/metadata/vital_metadata_supp.csv"),
                      row.names = 1) 
vital_supp_margin = read.csv(paste0(IMiC_path, "IMiC/Data/VITAL/metadata/vital_metadata_supp2.csv"),
                             row.names = 1) 
child_supp = read.csv(paste0(IMiC_path, "IMiC/Data/CHILD/metadata/child_metadata_supp.csv"),
                      row.names = 1) 
child_supp_margin = read.csv(paste0(IMiC_path, "IMiC/Data/CHILD/metadata/child_metadata_supp2.csv"),
                             row.names = 1) 
misame_traj_key = read.csv(paste0(IMiC_path, "IMiC/Data/MISAME/metadata/misame_processed_traj_key.csv"), row.names = 1)
vital_traj_key = read.csv(paste0(IMiC_path, "IMiC/Data/VITAL/metadata/vital_processed_traj_key.csv"), row.names = 1)

# Raw metadata (for temporal growth analyses)
misame_metadata = read.delim(paste0(IMiC_path, "IMiC/Data/MISAME/metadata/MISAME_3_IMiC_analysis.csv"), sep=",")
vital_metadata = read.delim(paste0(IMiC_path, "IMiC/Data/VITAL/metadata/VITAL_Lactation_IMiC_analysis (2).csv"), sep=",")

#Joining loaded data and metadata 
misame_full_meta = full_join(full_join(full_join(misame_full_BM_df, 
                                                 misame_supp %>% dplyr::rename(SubjectID=SUBJIDO)),
                                       misame_supp_margin), misame_traj_key)

Misame_everything = full_join(full_join(full_join(misame_full_meta, 
                                                   misame_targeted %>% 
                                                     tibble::rownames_to_column('SampleID')),
                                         misame_104_untargeted_feats),
                               misame_macronut.csv)  %>%
  mutate(Postnatal_BEP=as.factor(as.logical(BEP))) %>%
  filter(!is.na(SampleID)) %>% filter(!is.na(Timepoint)) %>%
  mutate(TP_label = factor(case_when(
    Timepoint==1 ~ "14-21 Days",
    Timepoint==2 ~ "1-2 Months",
    Timepoint==3 ~ "3-4 Months",
    TRUE ~ NA
  ), levels = c("14-21 Days", "1-2 Months", "3-4 Months")))

vital_full_meta = full_join(full_join(full_join(vital_full_BM_df, 
                                                vital_supp %>% dplyr::rename(SubjectID=SUBJIDO)),
                                      vital_supp_margin), 
                            vital_traj_key %>% dplyr::rename(SUBJID=SubjectID))

Vital_everything = full_join(full_join(full_join(vital_full_meta, 
                                                  vital_targeted %>% 
                                                    tibble::rownames_to_column('SampleID')),
                                        vital_104_untargeted_feats),
                              vital_macronut.csv)  %>%
  mutate(Postnatal_BEP=as.factor(as.logical(BEP))) %>%
  filter(!is.na(SampleID)) %>% filter(!is.na(Timepoint)) %>%
  mutate(TP_label = factor(case_when(
    Timepoint==1 ~ "40 Days",
    Timepoint==2 ~ "56 Days",
    TRUE ~ NA
  ), levels = c("40 Days", "56 Days")))

child_full_meta = full_join(full_join(child_full_BM_df, 
                                      child_supp %>% dplyr::rename(SubjectID=SUBJIDO), by = "SubjectID"),
                            child_supp_margin)

Child_everything = full_join(full_join(child_full_meta, 
                                        child_targeted %>% 
                                          tibble::rownames_to_column('SampleID')),
                              child_104_untargeted_feats) %>%
  left_join(child_macronut.csv) %>% filter(!is.na(Timepoint)) %>%
  mutate(TP_label = case_when(Timepoint==1 ~ "3 Months", TRUE~NA))

#Loading cross-study features from Machine learning analysis
cross_WAZ_untar = read.csv(paste0(IMiC_path, "IMiC/Code/machine_learning/metabolite_list_12.4.24.csv")) %>%
  filter(type=="growth") %>% pull(metabolite)
cross_BEP_untar = read.csv(paste0(IMiC_path, "IMiC/Code/machine_learning/metabolite_list_12.4.24.csv")) %>%
  filter(type=="BEP") %>% pull(metabolite)

targeted_feats = colnames(misame_targeted)[(colnames(misame_targeted) %in% colnames(vital_targeted)) & (colnames(misame_targeted) %in% colnames(child_targeted))]
untargeted_feats = colnames(misame_104_untargeted_feats)[-1]

#Loading Joint-RPCA Biocrates results
Misame_cov = read.csv(paste0(IMiC_path, "IMiC/Code/Joint-RPCA/output/misame_feature_cov_full_table_final.csv"), row.names=1, check.names = FALSE)
Vital_cov = read.csv(paste0(IMiC_path, "IMiC/Code/Joint-RPCA/output/vital_feature_cov_full_table_original_IDs.csv"), row.names=1,
                     check.names = FALSE)
rownames(Vital_cov) = str_replace(str_replace(rownames(Vital_cov), "P ", "P"),"LNFPIII", "LNFP III") # fixing label discrepancies

Misame_ord_biocrates_raw = read_excel(paste0(IMiC_path, "IMiC/Code/Joint-RPCA/output/misame_feature_loadings_final.xlsx"), sheet ="Biocrates") 
Misame_ord_biocrates = Misame_ord_biocrates_raw %>%
  tibble::column_to_rownames(colnames(Misame_ord_biocrates_raw)[1])

Vital_ord_biocrates_raw = read_excel(paste0(IMiC_path, "IMiC/Code/Joint-RPCA/output/vital_feature_loadings_final.xlsx"), sheet ="Biocrates") 
Vital_ord_biocrates = Vital_ord_biocrates_raw %>%
  tibble::column_to_rownames(colnames(Vital_ord_biocrates_raw)[1])

#Machine learning features identified via SHAP
targeted_cross_WAZ = c('Asp','FLNH','PC.ae.C34.0','g-tocopherol','lysoPC.a.C18.0','HipAcid','B2','Se','SM.C26.0',"6'SL",'SM.C18.1','LNFP III','Thr','AABA','HexCer.d18.1.18.1.','Fuc','Mn','C5','Ind.SO4','Phe','PA','LNnT','TG.17.2_38.5.','B6','B1','p.Cresol.SO4','Met.SO','TG.20.3_32.2.','Ca')
targeted_cross_BEP = c('Leu',"3'SL",'X3.Met.His', 'Lys','HipAcid','TG.16.1_36.1.','B2','TG.18.2_36.0.','LSTb',"6'SL",'Creatinine','B3','Mn','PA','a-tocopherol','K','Bio','B1','p.Cresol.SO4','Cit')

#Month 3 growth velocity calculations
misame_weight_diff_df_3m = left_join(misame_metadata, inner_join(
  misame_metadata %>% filter(VISIT=="Delivery", is.na(VISIT_R_FL)) %>%dplyr::select(SUBJID, WTKG) %>% dplyr::rename(wt_0=WTKG),
  misame_metadata %>% filter(VISIT=="Post Natal FU M03", is.na(VISIT_R_FL)) %>%dplyr::select(SUBJID, WTKG, AGEDAYS) %>% dplyr::rename(wt_3=WTKG, AGE=AGEDAYS))) %>%
  mutate(delta_weight_3mo = wt_3-wt_0,
         growth_vel_3mo=(1000*log(wt_3/wt_0))/AGE) %>%dplyr::select(SUBJIDO, growth_vel_3mo) %>%
  dplyr::rename(SubjectID=SUBJIDO)

vital_weight_diff_df_3m = left_join(vital_metadata, inner_join(
  vital_metadata %>% filter(VISIT=="Baseline", is.na(VISIT_R_FL)) %>%dplyr::select(SUBJID, WTKG) %>%dplyr::rename(wt_0=WTKG),
  vital_metadata %>% filter(VISIT=="Follow up month 3", is.na(VISIT_R_FL)) %>%dplyr::select(SUBJID, WTKG, AGEDAYS) %>%dplyr::rename(wt_3=WTKG, AGE=AGEDAYS))) %>%
  mutate(delta_weight_3mo = wt_3-wt_0,
         growth_vel_3mo=(1000*log(wt_3/wt_0))/AGE) %>%dplyr::select(SUBJIDO, growth_vel_3mo) %>%
  dplyr::rename(SubjectID=SUBJIDO) 

#Load Proteomics Data
misame_protein_meta = read.csv(paste0(IMiC_path, "IMiC/Data/Proteomics/MISAME/Metadata/PBL_MISAME_Protein_dictionary.csv"), row.names = 1)
misame_proteomics = read.csv(paste0(IMiC_path, "IMiC/Data/Proteomics/MISAME/Research_Data/PBL_MISAME_DIANN_Protein.csv"), row.names = 1)
rownames(misame_proteomics) <- str_remove(rownames(misame_proteomics), "_lama")
Misame_NA_list = c()
for (each_feat in colnames(misame_proteomics)) {
  each_col = misame_proteomics[[each_feat]]
  NA_perc = length(each_col[is.na(each_col)])/length(each_col)
  # new_name = misame_protein_meta %>% filter(ProteinIds==each_feat) %>% pull(Gene.Name) 
  if (NA_perc > 0.5) { # exclude if percent NA is greater than 50%
    Misame_NA_list = c(Misame_NA_list, each_feat)
    #} else if (is.na(new_name)) { 
    #  Misame_NA_list = c(Misame_NA_list, each_feat) # also exclude if gene name is NA
  } 
}
misame_proteomics_filt = misame_proteomics %>% dplyr::select(-Misame_NA_list) %>%
  tibble::rownames_to_column("SampleID")

# misame_protein_meta = misame_protein_meta %>%
#   dplyr::mutate(Gene.Name = ifelse(is.na(Gene.Name), paste0("Unnamed_", ProteinIds), Gene.Name))

misame_proteomics_filt_genes = misame_proteomics_filt %>% tibble::column_to_rownames("SampleID")
colnames(misame_proteomics_filt_genes) = misame_protein_meta$Gene.Name[match(colnames(misame_proteomics_filt_genes), misame_protein_meta$ProteinIds)]

vital_proteomics = read.csv(paste0(IMiC_path, "IMiC/Data/Proteomics/VITAL/Research_Data/PBL_VITAL_L_DIANN_Protein.csv"), row.names = 1)
vital_protein_meta = read.csv(paste0(IMiC_path, "IMiC/Data/Proteomics/VITAL/Metadata/PBL_VITAL_L_Protein_dictionary.csv"), row.names = 1)

# vital_protein_meta = vital_protein_meta %>%
#   dplyr::mutate(Gene.Name = ifelse(is.na(Gene.Name), paste0("Unnamed_", ProteinIds), Gene.Name))

Vital_NA_list = c()
for (each_feat in colnames(vital_proteomics)) {
  each_col = vital_proteomics[[each_feat]]
  NA_perc = length(each_col[is.na(each_col)])/length(each_col)
  # new_name = vital_protein_meta %>% filter(ProteinIds==each_feat) %>% pull(Gene.Name) 
  if (NA_perc > 0.5) { # exclude if percent NA is greater than 50%
    Vital_NA_list = c(Vital_NA_list, each_feat)
    #} else if (is.na(new_name)) { 
    #  Vital_NA_list = c(Vital_NA_list, each_feat) # also exclude if gene name is NA
  }
}
vital_proteomics_filt = vital_proteomics %>% dplyr::select(-Vital_NA_list) %>%
  tibble::rownames_to_column("SampleID")

vital_proteomics_filt_genes = vital_proteomics_filt %>% tibble::column_to_rownames("SampleID")
colnames(vital_proteomics_filt_genes) = vital_protein_meta$Gene.Name[match(colnames(vital_proteomics_filt_genes), vital_protein_meta$ProteinIds)]

Misame_proteome_ready = left_join(misame_full_BM_df, left_join(misame_targeted %>% tibble::rownames_to_column('SampleID'),
                                                                misame_proteomics_filt))
Vital_proteome_ready = left_join(vital_full_BM_df, left_join(vital_targeted %>% tibble::rownames_to_column('SampleID'),
                                                              vital_proteomics_filt))

# Adding Na/K ratio
Misame_proteome_ready = Misame_proteome_ready %>%
  #left_join(Misame_everything %>% dplyr::select(SampleID, Protein, Fat)) %>%
  mutate(Na_K_Ratio = Na/K)
Vital_proteome_ready = Vital_proteome_ready %>%
  #left_join(Vital_everything %>% dplyr::select(SampleID, Protein, Fat)) %>%
  mutate(Na_K_Ratio = Na/K)

Misame_everything = Misame_everything %>% mutate(Na_K_Ratio = Na/K) %>%
  dplyr::rename("Total_Protein"="Protein") %>%
  rowwise() %>% mutate(Total_HMO = sum(c_across(X2.FL:DSLNH))) %>% ungroup()
Vital_everything = Vital_everything %>% mutate(Na_K_Ratio = Na/K) %>%
  dplyr::rename("Total_Protein"="Protein") %>%
  rowwise() %>% mutate(Total_HMO = sum(c_across(X2.FL:DSLNH))) %>% ungroup()
