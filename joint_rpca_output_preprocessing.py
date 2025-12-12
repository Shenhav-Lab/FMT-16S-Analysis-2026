import numpy as np
import pandas as pd
import glob

def misame_meta_add_cols(misame_metadata):
    
    #add wide WAZ margin
    misame_bottom_limit = -1
    misame_top_limit = 0
    misame_metadata['WAZ_M04_wide_margin_pub'] = misame_metadata['WAZ_M04'].apply(
        lambda x: '<-1' if x<misame_bottom_limit else ('>0' if x>misame_top_limit else np.nan))
    #create new BEP labels
    misame_metadata['BEP_pub'] = misame_metadata['BEP'].apply(lambda x: 'Yes' if x==1 else 'No')
    
    return misame_metadata

def vital_meta_add_cols(vital_metadata):
    
    #add wide WAZ margin
    vital_bottom_limit = -2
    vital_top_limit = -1
    vital_metadata['WAZ_M04_wide_margin_pub'] = vital_metadata['WAZ_M04'].apply(
        lambda x: '<-2' if x<vital_bottom_limit else ('>-1' if x>vital_top_limit else np.nan))
    #create new BEP labels
    vital_metadata['BEP_pub'] = vital_metadata['BEP'].apply(lambda x: 'Yes' if x==1 else 'No')
    
    return vital_metadata

def load_misame_raw_tables(dir_path='../output/all_tps'):
    
    tables_misame_all = {}
    for table_ in glob.glob(dir_path+'/*_raw.csv'):
        if any(x in table_ for x in ['vital','child','micro_all','metadata']):
            continue
        table_id = table_.split('_')[2]
        tables_misame_all[table_id] = pd.read_csv(table_, index_col=0)
    
    tables_misame_all['untargeted_sapient'] = pd.read_csv(dir_path+'/misame_untargeted_sapient.csv', index_col=0)

    return tables_misame_all

def load_vital_raw_tables(dir_path='../output/all_tps'):
    
    tables_vital_all = {}
    for table_ in glob.glob(dir_path+'/*_raw.csv'):
        if any(x in table_ for x in ['misame','child','micro_all','metadata']):
            continue
        table_id = table_.split('_')[2]
        tables_vital_all[table_id] = pd.read_csv(table_, index_col=0)
    
    tables_vital_all['untargeted_sapient'] = pd.read_csv(dir_path+'/vital_untargeted_sapient.csv', index_col=0)

    return tables_vital_all

def get_shared_cohort_features(mod_list, misame_rankings, vital_rankings):

    shared_num = {}
    shared_denom = {}

    for mod in mod_list:

        mis_mod_ = misame_rankings[mod]
        vit_mod_ = vital_rankings[mod]
        
        ##numerators
        mis_num_ = mis_mod_[mis_mod_['Log_Ratio_Annot']=='Num'].index
        vit_num_ = vit_mod_[vit_mod_['Log_Ratio_Annot']=='Num'].index
        shared_num[mod] = list(set(mis_num_).intersection(vit_num_))

        ##denominators
        mis_den_ = mis_mod_[mis_mod_['Log_Ratio_Annot']=='Denom'].index
        vit_den_ = vit_mod_[vit_mod_['Log_Ratio_Annot']=='Denom'].index
        shared_denom[mod] = list(set(mis_den_).intersection(vit_den_))

    return shared_num, shared_denom