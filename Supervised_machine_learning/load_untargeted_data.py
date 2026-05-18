## Load data for IMiC untargeted metabolomics
# python script for imputation and normalization 
from ml_functions import data_preprocessing
import argparse

import sys
import os
import re
import pickle
import random
import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.model_selection import (train_test_split, LeaveOneOut)
from sklearn import metrics
from sklearn.metrics import (roc_curve, auc, confusion_matrix, f1_score)
from collections import Counter
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns

import scipy

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
import statsmodels.api as sm

from venny4py.venny4py import *

# parse args
parser = argparse.ArgumentParser(description='-d Dropbox_path')
parser.add_argument('-d','--directory', help='Dropbox path', required=True)
args = vars(parser.parse_args())
Dropbox_path = args['directory']

# Read metadata
print("loading metadata")
misame_full_BM_df_filt=pd.read_csv(os.path.join(Dropbox_path, "IMiC/Data/MISAME/metadata/misame_processed_metadata.tsv"), sep='\t')
misame_traj_df=pd.read_csv(os.path.join(Dropbox_path, "IMiC/Data/MISAME/metadata/misame_processed_traj_key.csv"), sep=',') # from growth_traj_MISAME.Rmd
misame_full_BM_df_filt=pd.merge(misame_full_BM_df_filt, misame_traj_df, on = 'SubjectID', how = 'left') 

vital_full_BM_df_filt=pd.read_csv(os.path.join(Dropbox_path, "IMiC/Data/VITAL/metadata/vital_processed_metadata.tsv"), sep='\t')
vital_traj_df=pd.read_csv(os.path.join(Dropbox_path, "IMiC/Data/VITAL/metadata/vital_processed_traj_key.csv"), sep=',') # from growth_traj_VITAL.Rmd
vital_traj_df=vital_traj_df.rename(columns={'SubjectID':'SUBJID'})
vital_full_BM_df_filt=pd.merge(vital_full_BM_df_filt, vital_traj_df[['SUBJID', 'WAZ_traj']], left_on='SUBJID',right_on='SUBJID', how='left')

# CHILD metadata (berkeley anthropometrics)
#child_metadata=pd.read_csv(os.path.join(Dropbox_path, "IMiC/Data/CHILD/metadata/CHILD_IMiC_analysis.csv"), sep=',')
child_metadata=pd.read_csv(os.path.join(Dropbox_path, "IMiC/Data/CHILD/metadata/child_processed_metadata.tsv"), sep='\t')
child_metadata.rename(columns={'BMID': 'SampleID', 'SUBJIDO': 'SubjectID'}, inplace=True)
child_metadata=child_metadata[~child_metadata["SampleID"].isna()]
child_metadata["Timepoint"]=3
Child_month3_anthro=berkeley_anthro[(berkeley_anthro['STUDYID']=='CHILD') & berkeley_anthro['VISIT'].str.endswith("3 Months FU P00")]
Child_month3_anthro.rename(columns={'WAZ': 'WAZ_M03', 'HAZ': 'HAZ_M03', 'WHZ': 'WHZ_M03', 'SUBJIDO': 'SubjectID'}, inplace=True)
Child_month3_anthro=Child_month3_anthro.assign(WAZ_M03_bins=np.where(np.isnan(Child_month3_anthro['WAZ_M03']), 
                                                                            np.nan, np.where(Child_month3_anthro['WAZ_M03']>-1, 0, 1)),
                                               HAZ_M03_bins=np.where(np.isnan(Child_month3_anthro['HAZ_M03']), 
                                                                            np.nan, np.where(Child_month3_anthro['HAZ_M03']>-1, 0, 1)),
                                               WHZ_M03_bins=np.where(np.isnan(Child_month3_anthro['WHZ_M03']), 
                                                                            np.nan, np.where(Child_month3_anthro['WHZ_M03']>-1, 0, 1)))

# Metabolomics data transformation
print("loading data")
if 'misame_preproc' not in globals():
    misame_raw=pd.read_csv(os.path.join(Dropbox_path, "IMiC/Data/Sapient_Box_Data/Raw/MISAME3_rLC_mtb_raw_data_021224.csv"), sep=',', index_col=0)
    misame_raw= misame_raw[misame_raw.index.isin(misame_full_BM_df_filt['SampleID'])]
    missingness_file = os.path.join(Dropbox_path, "IMiC/Data/MISAME/Sapient_missingness.csv")
    if not os.path.exists(missingness_file):
        pd.DataFrame(pd.isna(misame_raw).mean(), columns = ['Missingness']).to_csv(missingness_file)
    misame_array = misame_raw.values
    random.seed(5)
    #draw from unif distribution with low=min(table)/10 and high=min(table)
    zero_len = len(misame_array[pd.isna(misame_array)])
    print(f'MISAME untargeted missing values: {zero_len/misame_raw.size}')
    min_value = np.nanmin(misame_array)
    rand_array = np.random.uniform(low=min_value/10, high=min_value, size=zero_len)
    misame_array[pd.isna(misame_array)] = rand_array
    misame_preproc=pd.DataFrame(misame_array, index=misame_raw.index, columns=misame_raw.columns).reset_index()
print(misame_preproc.shape)
if 'vital_preproc' not in globals():
    vital_raw=pd.read_csv(os.path.join(Dropbox_path, "IMiC/Data/Sapient_Box_Data/Raw/CHILD_ELICIT_VITAL_rLC_mtb_raw_data_021224.csv"), sep=',', index_col=0)
    vital_raw= vital_raw[vital_raw.index.isin(vital_full_BM_df_filt['SampleID'])]
    missingness_file = os.path.join(Dropbox_path, "IMiC/Data/VITAL/Sapient_missingness.csv")
    if not os.path.exists(missingness_file):
        pd.DataFrame(pd.isna(vital_raw).mean(), columns = ['Missingness']).to_csv(missingness_file)
    vital_array = vital_raw.values
    random.seed(100)
    #draw from unif distribution with low=min(table)/10 and high=min(table)
    zero_len = len(vital_array[pd.isna(vital_array)])
    print(f'Mumta-LW untargeted missing values: {zero_len/vital_raw.size}')
    min_value = np.nanmin(vital_array)
    rand_array = np.random.uniform(low=min_value/10, high=min_value, size=zero_len)
    vital_array[pd.isna(vital_array)] = rand_array
    vital_preproc=pd.DataFrame(vital_array, index=vital_raw.index, columns=vital_raw.columns).reset_index()
print(vital_preproc.shape)
if 'child_preproc' not in globals():
    child_raw=pd.read_csv(os.path.join(Dropbox_path, "IMiC/Data/Sapient_Box_Data/Raw/CHILD_ELICIT_VITAL_rLC_mtb_raw_data_021224.csv"), sep=',', index_col=0)
    child_raw.index=child_raw.index.map(lambda index: re.sub(r'-\d+$', '', index))
    child_raw= child_raw[child_raw.index.isin(child_metadata['SampleID'])]
    missingness_file = os.path.join(Dropbox_path, "IMiC/Data/CHILD/Sapient_missingness.csv")
    if not os.path.exists(missingness_file):
        pd.DataFrame(pd.isna(child_raw).mean(), columns = ['Missingness']).to_csv(missingness_file)
    child_array = child_raw.values
    random.seed(200)
    #draw from unif distribution with low=min(table)/10 and high=min(table)
    zero_len = len(child_array[pd.isna(child_array)])
    print(f'CHILD untargeted missing values: {zero_len/child_raw.size}')
    min_value = np.nanmin(child_array)
    rand_array = np.random.uniform(low=min_value/10, high=min_value, size=zero_len)
    child_array[pd.isna(child_array)] = rand_array
    child_preproc=pd.DataFrame(child_array, index=child_raw.index, columns=child_raw.columns).reset_index()
print(child_preproc.shape)

print("reading keys")
# Read ID key files
cross_study_key_original=pd.read_csv(os.path.join(Dropbox_path, "IMiC/Data/MISAME/key/IMiC_alignment.csv"), sep = ',')
cross_study_key_new=pd.read_excel(os.path.join(Dropbox_path, "IMiC/Data/MISAME/IMiC_Alignment_Updated_April232024.xlsx"), sheet_name='Alignment')
# Read MS2 Data
Kim_Misame=pd.read_excel(os.path.join(Dropbox_path, "IMiC/Data/Sapient_Biomarkers/MISAME3_Sig_Biomarkers.xlsx"))
Kim_VitalChildElicit=pd.read_excel(os.path.join(Dropbox_path, "IMiC/Data/Sapient_Biomarkers/CHILD_ELICIT_VITAL_Sig_Biomarkers.xlsx"))
# Subsetting the original key: remove NAs, sort by rt_difference, remove duplicates from misame and vital ID columns with higher rt_difference 
cross_study_key_original_besthits=pd.read_csv(os.path.join(Dropbox_path, "IMiC/Data/MISAME/key/IMiC_alignment.csv"), sep = ',')
cross_study_key_original_besthits=cross_study_key_original_besthits.dropna(subset=['mtb_id_MISAME3', 'mtb_id_CHILD_ELICIT_VITAL']).sort_values(by='rt_difference')[['mtb_id_MISAME3', 'mtb_id_CHILD_ELICIT_VITAL']]
cross_study_key_original_besthits=cross_study_key_original_besthits.drop_duplicates(subset='mtb_id_MISAME3', keep="first")
cross_study_key_original_besthits=cross_study_key_original_besthits.drop_duplicates(subset='mtb_id_CHILD_ELICIT_VITAL', keep="first")
cross_study_key_original_besthits
# Identifying 1-1 overlap IDs from original key
Misame_MisVit2_common_IDs=cross_study_key_original_besthits['mtb_id_MISAME3'].tolist()
Vital_MisVit2_common_IDs=cross_study_key_original_besthits['mtb_id_CHILD_ELICIT_VITAL'].tolist()
MisVit2_rename_dict = dict(zip(cross_study_key_original_besthits['mtb_id_CHILD_ELICIT_VITAL'], cross_study_key_original_besthits['mtb_id_MISAME3']))
misame_1to1_preproc=misame_preproc[['sample_ID'] + Misame_MisVit2_common_IDs]
vital_1to1_preproc=vital_preproc[['sample_ID'] + Vital_MisVit2_common_IDs].rename(columns=MisVit2_rename_dict)
child_1to1_preproc=child_preproc[['sample_ID'] + Vital_MisVit2_common_IDs].rename(columns=MisVit2_rename_dict)

misame_full_BM_df_filt=misame_full_BM_df_filt.assign(WAZ_M04_bins=np.where(np.isnan(misame_full_BM_df_filt['WAZ_M04']), np.nan,
                                                                            np.where(misame_full_BM_df_filt['WAZ_M04']>-1, 0, 1)),
                                                    HAZ_M04_bins=np.where(np.isnan(misame_full_BM_df_filt['HAZ_M04']), np.nan,
                                                                            np.where(misame_full_BM_df_filt['HAZ_M04']>-1, 0, 1)),
                                                    WHZ_M04_bins=np.where(np.isnan(misame_full_BM_df_filt['WHZ_M04']), np.nan,
                                                                            np.where(misame_full_BM_df_filt['WHZ_M04']>-1, 0, 1)),
                                                    BAZ_M04_bins=np.where(np.isnan(misame_full_BM_df_filt['BAZ_M04']), np.nan,
                                                                            np.where(misame_full_BM_df_filt['BAZ_M04']>-1, 0, 1)),
                                                    MUAZ_M04_bins=np.where(np.isnan(misame_full_BM_df_filt['MUAZ_M04']), np.nan,
                                                                            np.where(misame_full_BM_df_filt['MUAZ_M04']>-1, 0, 1)),
                                                    HCIRCM_M04_bins=np.where(np.isnan(misame_full_BM_df_filt['HCIRCM_M04']), np.nan,
                                                                            np.where(misame_full_BM_df_filt['HCIRCM_M04']>-1, 0, 1)))
cols=(misame_full_BM_df_filt.columns.tolist())
col_list=['SubjectID', 'WAZ_M04', 'HAZ_M04', 'WHZ_M04', 'WAZ_traj', 'ARMCD', 'BIRTHWT', 'BIRTHLEN', 'MAGE', 'NPERSON', 'DVSEASON', 'TermType', 'MHTCM_Enroll', 
          'Low_birth_weight_binary', 'Premature_binary', 'Small_vulnerable_newborns_binary', 'MAGE_binary_bins', 'H20_type', 'WAZ_M04_binary_bins',
          'MBMI_AC4_binary_bins', 'MBMI_M01_binary_bins', 'MBMI_M04_binary_bins', 'MBMI_Deliv_binary_bins', 'MMUACCM_M04_binary_bins', 'MMUACCM_Enroll', 'MHTCM_binary_bins',
          'Mat_edu_binary_bins', 'Summer', 'any_HOSP', 'any_DIARR', 'any_FEVER', 'any_COUGH', 'any_VOMIT', 'any_SICK', 'any_inf_anemic', 'any_mat_anemic', 'growth_faltering', 'HCIRCM_M04_bins', 'MUAZ_M04_bins',
          'WAZ_M4_M0_falter_binary_bins', 'WAZ_M4_M0_thrive_binary_bins']


misame_meta_for_ML=misame_full_BM_df_filt[col_list].drop_duplicates()
misame_meta_for_ML=misame_meta_for_ML.assign(DVSEASON=np.where(misame_meta_for_ML['DVSEASON']=="Rainy Season", 0, 1),
                                             Prematurity=np.where(misame_meta_for_ML['TermType']=="Premature", 1, 0),
                                             TermType=np.where(misame_meta_for_ML['TermType']=="Full Term", 0, 
                                                                       np.where(misame_meta_for_ML['TermType']=="Early Term", 1, 2)),
                                             Traj_WAZ=np.where(misame_meta_for_ML['WAZ_traj']==5, 1, 
                                                                np.where(misame_meta_for_ML['WAZ_traj']==3, 0, np.nan)),
                                             Group_3=np.where(misame_meta_for_ML['WAZ_traj']==3, 1, 
                                                                np.where(misame_meta_for_ML['WAZ_traj']!=3, 0, np.nan)))
misame_meta_for_ML = misame_meta_for_ML.assign(postnatal_BEP=np.where(misame_meta_for_ML['ARMCD'].isin([2, 4]), 1, 0))
misame_meta_for_ML.head(50)


vital_full_BM_df_filt=vital_full_BM_df_filt.assign(WAZ_M04_bins=np.where(np.isnan(vital_full_BM_df_filt['WAZ_M04']), np.nan,
                                                                            np.where(vital_full_BM_df_filt['WAZ_M04']>-1, 0, 1)),
                                                    HAZ_M04_bins=np.where(np.isnan(vital_full_BM_df_filt['HAZ_M04']), np.nan,
                                                                            np.where(vital_full_BM_df_filt['HAZ_M04']>-1, 0, 1)),
                                                    WHZ_M04_bins=np.where(np.isnan(vital_full_BM_df_filt['WHZ_M04']), np.nan,
                                                                            np.where(vital_full_BM_df_filt['WHZ_M04']>-1, 0, 1)),
                                                    BAZ_M04_bins=np.where(np.isnan(vital_full_BM_df_filt['BAZ_M04']), np.nan,
                                                                            np.where(vital_full_BM_df_filt['BAZ_M04']>-1, 0, 1)),
                                                    MUAZ_M04_bins=np.where(np.isnan(vital_full_BM_df_filt['MUAZ_M04']), np.nan,
                                                                            np.where(vital_full_BM_df_filt['MUAZ_M04']>-1, 0, 1)))
col_list=['SubjectID', 'WAZ_M04', 'HAZ_M04', 'WHZ_M04', 'WAZ_traj', 'Azithromycin_binary', 'ARMCD', 'BIRTHWT', 'BIRTHLEN', 'MAGE', 'NPERSON', 'DVSEASON', 'TermType', 'MHTCM', 
          'Low_birth_weight_binary', 'Premature_binary', 'Small_vulnerable_newborns_binary', 'MAGE_binary_bins', 'H20_type', 'WAZ_M04_binary_bins',
          'MBMI_M01_binary_bins', 'MBMI_M02_binary_bins', 'MBMI_M03_binary_bins', 'MBMI_M04_binary_bins', 'MMUACCM_M04_binary_bins', 'MHTCM_binary_bins',
          'Mat_edu_binary_bins', 'Monsoon', 'any_HOSP', 'any_DIARR', 'any_FEVER', 'any_COUGH', 'any_VOMIT', 'any_SICK', 'any_inf_anemic', 'any_mat_anemic', 'growth_faltering', 'MUAZ_M04_bins',
          'WAZ_M4_M0_falter_binary_bins', 'WAZ_M4_M0_thrive_binary_bins']
vital_meta_for_ML=vital_full_BM_df_filt[col_list].drop_duplicates()
vital_meta_for_ML=vital_meta_for_ML.assign(postnatal_BEP=np.where(vital_meta_for_ML['ARMCD'].isin([2, 3]), 1, 0),
                                           Prematurity=np.where(vital_meta_for_ML['TermType']=="Premature", 1, 0),
                                           TermType=np.where(vital_meta_for_ML['TermType']=="Full Term", 0,
                                                             np.where(vital_meta_for_ML['TermType']=="Early Term", 1, 2)),
                                           WAZ_traj_bins=np.where(vital_meta_for_ML['WAZ_traj']==4, 1, 0))
vital_meta_for_ML

#### New metadata steps for CHILD 
col_list=['SubjectID', 'BIRTHWT', 'BIRTHLEN', 'MAGE', 'NPERSON', 'DVSEASON', 'GAGEBRTH', 'HCAZ_M03_binary_bins', 'SampleID']
child_meta_for_ML=pd.merge(Child_month3_anthro, child_metadata[col_list].drop_duplicates())
child_meta_for_ML=child_meta_for_ML.assign(TermType=np.where(child_meta_for_ML['GAGEBRTH']<259, "Premature", np.where(child_meta_for_ML['GAGEBRTH']<273, "Early Term", "Full Term")))
child_meta_for_ML=child_meta_for_ML.assign(Prematurity=np.where(child_meta_for_ML['TermType']=="Premature", 1, 0))
child_meta_for_ML
print("reading Sapient top metabolites")
Kim_Misame=pd.read_excel(os.path.join(Dropbox_path, "IMiC/Data/Sapient_Biomarkers/MISAME3_Sig_Biomarkers.xlsx"))
waz_cols=['weight_az1_1421','weight_az1_pn34','weight_az2_1421','c_waz3_1421','c_waz3_pn12']
Kim_Misame_waz_set=set(Kim_Misame[Kim_Misame['outcome_time'].isin(waz_cols)]['feature_label'].tolist())
len(Kim_Misame_waz_set)
Kim_Misame_bep_set=set(Kim_Misame[Kim_Misame['analysis']=='MISAME3-bep']['feature_label'].tolist())
len(Kim_Misame_bep_set)
Kim_Vital=pd.read_excel(os.path.join(Dropbox_path, "IMiC/Data/Sapient_Biomarkers/CHILD_ELICIT_VITAL_Sig_Biomarkers.xlsx"))
waz_cols=['WAZ']
Kim_Vital_waz_set=set(Kim_Vital[Kim_Vital['outcome'].isin(waz_cols)]['feature_label'].tolist())
len(Kim_Vital_waz_set)
Kim_Vital_bep_set=set(Kim_Vital[Kim_Vital['analysis']=='VITAL-bep']['feature_label'].tolist())
len(Kim_Vital_bep_set)
