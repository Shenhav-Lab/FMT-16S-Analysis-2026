### functions for ML pipelines ###

# Loading libraries
import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import (roc_curve, auc, confusion_matrix, f1_score, precision_recall_curve)
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import RocCurveDisplay
from scipy.signal import savgol_filter
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score


### PREPROCESSING DATA ###
# convert timepoints to individual features
def timepoint_col_distribute(input_counts):
    # This function takes as input a dataframe with columns: 
    # sample_ID, "SubjectID", "Timepoint", followed by feature columns
    # creates new df with one SubjectID per row, with columns in format TP<N>_<Feature_col>
    output_counts=pd.DataFrame()
    for each_subject in input_counts['SubjectID'].unique():
        subject_input_counts = input_counts[input_counts['SubjectID'] == each_subject]
        # if subject_input_counts.shape[0]==len(input_counts['Timepoint'].unique()):
        new_row=pd.DataFrame({'SubjectID': [each_subject]})
        for TP in input_counts['Timepoint'].unique():
            TP_cols=subject_input_counts[subject_input_counts['Timepoint']==TP].iloc[:, 3:]
            TP_cols.columns = ['TP' + str(TP) + '_' + column_name for column_name in TP_cols.columns]
            if TP_cols.shape[0]==0:
                TP_cols.loc[0, :] = np.nan
            new_row=new_row.set_axis(TP_cols.index).join(TP_cols)
        output_counts=pd.concat([output_counts, new_row])
    return output_counts

# impute, transform, normalize, or scale data
# Change to using custom transformer classes to fit with sklearn api for much cleaner system!
def data_preprocessing(df, mode='fit', params = None, impute=None, transform=None, normalize=None, scale=None):
    # expects features as cols and samples as rows
    # 'fit' on a dataset, save params to a var, and 'transform' the dataset, using the params you saved
    # fit returns params, transformed df
    # imputers assume NaN not 0 values
    # options:
    # mode: ['fit', 'transform']
    # impute: ['draw_uniform', 'knn', 'rf']
    # transform: ['log', 'power']
    # normalize: ['quantile_norm']
    # scale: ['unit', 'pareto', 'range', 'vast', 'level']
    # im pretty sure impute, transform should be first but im not confident about scale and normalize 

    # follows https://doi.org/10.1016/j.gendis.2023.04.018

    # Things to potenitally implement:
    # ways to process outliers?
    # scaling approaches, can add x-vast

    col_names = df.columns
    row_names = df.index

    # all the parameter options
    mode_options = ['fit', 'transform']
    impute_options = ['draw_uniform', 'knn', 'rf']
    transform_options = ['log', 'power']
    normalize_options = ['quantile_norm']
    scale_options = ['unit', 'pareto', 'range', 'vast', 'level']

    # test to make sure parameters are in options    
    if mode not in mode_options:
        raise Exception("not one of mode_options: "+", ".join(mode_options))
    
    if impute!=None:
        if impute not in impute_options:
            raise Exception("not one of impute options: "+", ".join(impute_options))

    if transform!=None:
        if transform not in transform_options:
            raise Exception("not one of transform options: "+", ".join(transform_options))
    
    if normalize!=None:
        if normalize not in normalize_options:
            raise Exception("not one of normalize options: "+", ".join(normalize_options))
    
    if scale!=None:
        if scale not in scale_options:
            raise Exception("not one of scale options: "+", ".join(scale_options))
    
    # imputation functions
    def draw_uniform(df, mode, min=None): # draws uniformly between the min value/10 and min value
        if mode=='fit':
            min_value = np.nanmin(df)
        if mode=='transform':
            min_value = min
        df = df.fillna(value=np.random.uniform(min_value/10, min_value))
        return(['uniform', min_value], df)
        
    def knn_imputer(df, mode, model=None): # imputes using knn with 3 neighbors
        if mode=='fit':
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=3, keep_empty_features=True) # play with number of neighbors?
            imputer.fit(df)
        if mode=='transform':
            imputer = model
        df = imputer.transform(df)
        return(['knn', imputer],pd.DataFrame(df))

        
    def rf_imputer(df, mode, model=None): # imputes using random forest, does not work with run_xgb_parallel.py
        if mode=='fit':
            from missforest.missforest import MissForest
            from sklearn.ensemble import RandomForestRegressor
            rgr = RandomForestRegressor()
            mf = MissForest(rgr)
            mf.fit(df)
        if mode=='transform':
            mf = model
        df = mf.transform(df)
        return(['rf', mf], df)

    # normalization functions
    def quantile_normalization(df, mode, means=None): # https://academic.oup.com/bioinformatics/article/19/2/185/372664  , nonparametric approach to normalize measured intensities from a single fluorophore to a common distribution
        df = df.T
        sorted_indices = np.argsort(df, axis=0)
        reverse_sorting = np.argsort(sorted_indices,axis=0)
        sorted = np.take_along_axis(df.to_numpy(), sorted_indices, axis=0) 
        if mode=='fit':
            row_means = np.mean(sorted, axis=1, keepdims=True)
        if mode=='transform':
            row_means = means
        arr_mean_assigned = row_means * np.ones_like(sorted)
        unsorted = np.take_along_axis(arr_mean_assigned, reverse_sorting, axis=0).T
        return(['qn', row_means], pd.DataFrame(unsorted))
    
    # scaling functions
    def unit_scaler(df, mode, means=None, sds=None): # standard scaler to N(0,1)
        if mode=='fit':
            means = []
            sds = []
            cols = np.empty(df.shape)
            for i, col_name in enumerate(df):
                col = df.iloc[:,i]
                mean = np.mean(col)
                means.append(mean)
                if np.std(col)==0:
                    sd = 1
                else:
                    sd = np.std(col)
                sds.append(sd)
                col = col.apply(lambda x: ((x-mean)/sd))
                cols[:,i] = col
            return(['unit', means, sds], cols)
        if mode=='transform':
            cols = np.empty(df.shape)
            for i, col_name in enumerate(df):
                col = df.iloc[:,i]
                mean = means[i]
                sd = sds[i]
                col = col.apply(lambda x: ((x-mean)/sd))
                cols[:,i] = col
            return(['unit', means, sds], cols)

    def pareto_scaler(df, mode, means=None, sqrt_sds=None): # like normal but scaled by sqrt(sd) instead of sd
        if mode=='fit':
            means = []
            sqrt_sds = []
            cols = np.empty(df.shape)
            for i, col_name in enumerate(df):
                col = df.iloc[:,i]
                mean = np.mean(col)
                means.append(mean)
                if np.sqrt(np.std(col))==0:
                    sqrt_sd = 1
                else:
                    sqrt_sd = np.sqrt(np.std(col))
                sqrt_sds.append(sqrt_sd)
                col = col.apply(lambda x: (x-mean)/sqrt_sd)
                cols[:,i] = col
            return(['pareto', means, sqrt_sds], cols)
        if mode=='transform':
            cols = np.empty(df.shape)
            for i, col_name in enumerate(df):
                col = df.iloc[:,i]
                mean = means[i]
                sqrt_sd = sqrt_sds[i]
                col = col.apply(lambda x: (x-mean)/sqrt_sd)
                cols[:,i] = col
            return(['pareto', means, sqrt_sds], cols)
    
    def range_scaler(df, mode, means = None, ranges = None): # scaled by the range of the data
        if mode=='fit':
            means = []
            ranges = []
            cols = np.empty(df.shape)
            for i, col_name in enumerate(df):
                col = df.iloc[:,i]
                mean = np.mean(col)
                means.append(mean)
                range = np.max(col)-np.min(col)
                ranges.append(range)
                col = col.apply(lambda x: (x-mean)/range)
                cols[:,i] = col
            return(['range', means, ranges], cols)
        if mode=='transform':
            cols = np.empty(df.shape)
            for i, col_name in enumerate(df):
                col = df.iloc[:,i]
                mean = means[i]
                range = ranges[i]
                col = col.apply(lambda x: (x-mean)/range)
                cols[:,i] = col
            return(['range', means, ranges], cols)    

    
    def vast_scaler(df, mode, means=None, sds=None): # standard scaler, then scaled by mean/sd
        if mode=='fit':
            means = []
            sds = []
            cols = np.empty(df.shape)
            for i, col_name in enumerate(df):
                col = df.iloc[:,i]
                mean = np.mean(col)
                means.append(mean)
                if np.std(col)==0:
                    sd = 1
                else:
                    sd = np.std(col)
                sds.append(sd)
                col = col.apply(lambda x: ((x-mean)/sd)*(mean/sd))
                cols[:,i] = col
            return(['vast', means, sds], cols)
        if mode=='transform':
            cols = np.empty(df.shape)
            for i, col_name in enumerate(df):
                col = df.iloc[:,i]
                mean = means[i]
                sd = sds[i]
                col = col.apply(lambda x: ((x-mean)/sd)*(mean/sd))
                cols[:,i] = col
            return(['vast', means, sds], cols)
        
    def level_scaler(df, mode, means=None): # scaled by mean
        if mode=='fit':
            means = []
            cols = np.empty(df.shape)
            for i, col_name in enumerate(df):
                col = df.iloc[:,i]
                mean = np.mean(col)
                means.append(mean)
                col = col.apply(lambda x: (x-mean)/mean)
                cols[:,i] = col
            return(['level', means], cols)
        if mode=='transform':
            cols = np.empty(df.shape)
            for i, col_name in enumerate(df):
                col = df.iloc[:,i]
                mean = means[i]
                col = col.apply(lambda x: (x-mean)/mean)
                cols[:,i] = col
            return(['level', means],cols)
        
    # fit the normalization models to the data AND transform it. Saves the params
    if mode=='fit':

        params = {}
        
        if impute in impute_options:
            if impute=='draw_uniform':
                params['impute'], df =  draw_uniform(df, mode)
            if impute=='knn':
                params['impute'], df = knn_imputer(df,mode)
            if impute=='rf':
                params['impute'], df = rf_imputer(df,mode)
        
        if transform in transform_options:
            params['transform'] = transform
            if transform=='log':
                df = np.log10(df, out=np.zeros_like(df, dtype=np.float64), where=(df!=0))
            if transform=='power':
                df = np.sqrt(df)
        
        if normalize in normalize_options:
            if normalize=='quantile_norm':
                params['normalize'], df = quantile_normalization(df, mode)
                
        if scale in scale_options:
            if scale=='unit': 
                # favors systematic changes, inflates importance of small metabolites
                params['scale'], df = unit_scaler(df, mode)
            if scale=='pareto': 
                # like unit but keeps data structure partially intact, sensitive to large fold changes
                params['scale'], df = pareto_scaler(df, mode)
            if scale=='range': 
                # compare metabs. relative to biological response range, sensitive to outliers
                params['scale'], df = range_scaler(df, mode)
            if scale=='vast': 
                # can focus on less fluctuated metabs., not effective for features with large variance
                params['scale'], df = vast_scaler(df, mode)
            if scale=='level': 
                # focus on relative changes, prone to inflate measurement error
                params['scale'],df = level_scaler(df, mode)

        
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        
        df.columns = col_names
        df.index = row_names
        
        return(params, df)

    # transform dataframe using params that were obtained in a previous run with 'fit'    
    if mode == 'transform':

        if 'impute' in params:
            impute = params['impute']
            if impute[0]=='uniform':
                min = impute[1]
                df = draw_uniform(df, mode=mode, min=min)[1]
            if impute[0]=='knn':
                model = impute[1]
                df = knn_imputer(df, mode=mode, model=model)[1]
            if impute[0]=='rf':
                model = impute[1]
                df = rf_imputer(df, mode=mode, model=model)[1]

        if 'transform' in params:
            transform = params['transform']
            if transform=='log':
                df = np.log10(df, out=np.zeros_like(df, dtype=np.float64), where=(df!=0))
            if transform=='power':
                df = np.sqrt(df)

        if 'normalize' in params:
            normalize = params['normalize']
            if normalize[0]=='qn':
                means = normalize[1]
                df = quantile_normalization(df, mode, means)[1]

        if 'scale' in  params:
            scale = params['scale']

            if scale[0]=='pareto':
                means = scale[1]
                sqrt_sds = scale[2]
                df = pareto_scaler(df, mode, means, sqrt_sds)[1]
            if scale[0]=='unit':
                means = scale[1]
                sds = scale[2]
                df = unit_scaler(df, mode, means, sds)[1]
            if scale[0]=='range':
                means = scale[1]
                ranges = scale[2]
                df = range_scaler(df, mode, means, ranges)[1]
            if scale[0]=='vast':
                means = scale[1]
                sds = scale[2]
                df = vast_scaler(df, mode, means, sds)[1]
            if scale[0]=='level':
                means = scale[1]
                df = level_scaler(df, mode, means)[1]

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        df.columns = col_names
        df.index = row_names

        return(df)

# get ratio between outcomes
def outcome_ratio(df, outcome):
    ratio = len(df[df[outcome]==0])/len(df[df[outcome]==1])
    print("Ratio outcome 0 / outcome 1:")
    print(ratio)
    return ratio


### MODEL TRAINING ###
# main ML model function for xgboost: 
# Runs XGBOOST model across all the monte-carlo CV seeds defined in seed_list and saves inputs/model/outputs/stats as a pickle file 
def run_gbm_bw(df, outcome_col, confounder_list, params_dict, seed_list, pickle_path, use_SMOTE, startover): 
    roc_scores = []
    f1_scores = []
    ber_scores = []
    if startover==True:
        output= {}
    else:
        if os.path.isfile(pickle_path):
            output=pd.read_pickle(pickle_path)
            for key in output:
                roc_scores.append(output[key]['AUC'])
                f1_scores.append(output[key]['F1'])
                ber_scores.append(output[key]['BER'])
            print("ROC AUC mean:"+str(np.mean(roc_scores)))
            print("ROC AUC median:"+str(np.median(roc_scores)))
            print("F1 median:"+str(np.median(f1_scores)))
            print("BER median:"+str(np.median(ber_scores)))
            return output
        elif os.path.isfile(pickle_path + '.temp'): 
            print("temporary files found")
            output=pd.read_pickle(pickle_path + '.temp')
        else:
            output= {}
        
    df=df.rename(columns={outcome_col: 'Outcome_bins'})
    for each_seed in seed_list:
        if each_seed in output: 
            print("results found for seed: " + str(each_seed))
        else:
            np.random.seed(each_seed)

            # SMOTE pipeline
            over = SMOTE(sampling_strategy=0.6, random_state=each_seed)
            under = RandomUnderSampler(sampling_strategy=0.7, random_state=each_seed)
            steps = [('over', over), ('under', under)]
            steps_under = [('under', under)]
            steps_over = [('over', over)]
            pipeline = Pipeline(steps=steps)

            # Train/Test split 80/20, stratified for outcome variable
            train, test = train_test_split(df, test_size=0.25, stratify=df['Outcome_bins']) 
            X_trainset=train.drop(columns=['Outcome_bins', 'SubjectID'])
            y_trainset=train['Outcome_bins']
            X_testset=test.drop(columns=['Outcome_bins', 'SubjectID'])
            y_testset=test['Outcome_bins']

            if len(confounder_list)>0:
                # Cross-Validated Confound Regression
                X_testset_feat=X_testset.drop(confounder_list, axis=1)
                X_testset_confounders=X_testset[confounder_list]
                X_trainset_feat=X_trainset.drop(confounder_list, axis=1)
                X_trainset_confounders=X_trainset[confounder_list]
                #   Building OLS with test data 
                model_fwl = sm.OLS(X_trainset_feat.values, X_trainset_confounders.values, missing="drop").fit()
                #   Subtract residuals from train
                ols_pred_values_train = model_fwl.predict(X_trainset_confounders.values)
                resids_train = X_trainset_feat.values-ols_pred_values_train
                resids_train_df = pd.DataFrame(resids_train, columns=X_trainset_feat.columns)
                X_trainset = resids_train_df
                #   Subtract residuals from test
                ols_pred_values_test = model_fwl.predict(X_testset_confounders.values)
                resids_test = X_testset_feat.values - ols_pred_values_test
                resids_test_df = pd.DataFrame(resids_test, columns=X_testset_feat.columns)
                X_testset = resids_test_df

            print("test and train shape:")
            print(X_trainset.shape)
            print(X_testset.shape)

            # Modeling: gradient boosting classifier from sklearn
            params_dict['random_state'] = each_seed
            tree_model=xgb.XGBClassifier(**params_dict) 

            # Resampling trainsets using SMOTE pipeline
            if (use_SMOTE==True):
                X_resampled, y_resampled = pipeline.fit_resample(X_trainset, y_trainset)
                tree_model.fit(X_resampled, y_resampled)
            else:
                tree_model.fit(X_trainset,y_trainset)

            # Prediction
            # tree_predictions=tree_model.predict(X_testset)

            # Evaluation
            y_pred_prob = tree_model.predict_proba(X_testset)[:, 1]
            y_pred = tree_model.predict(X_testset)

            f1=f1_score(y_testset, y_pred)
            fpr, tpr, thresholds = roc_curve(y_testset, y_pred_prob, pos_label=1)
            roc_auc = auc(fpr, tpr)
            # precision, recall, _ = precision_recall_curve(y_testset, y_pred_prob)
            # auc_pr = auc(recall, precision)
            conf_mat=confusion_matrix(y_testset, y_pred)
            ber = get_BER(conf_mat)
            
            print(conf_mat)
            print("ROC score for seed: {} = {}".format(each_seed, roc_auc))
            print("F1 score for seed: {} = {}".format(each_seed, f1))
            print("BER score for seed: {} = {}".format(each_seed, ber))

            each_output = {"Seed":each_seed, "Outcome":outcome_col, "X_testset":X_testset, 
                    "X_trainset":X_trainset, "y_testset":y_testset, "y_trainset":y_trainset, "Fit_model":tree_model, "AUC":roc_auc,
                    "Confusion_matrix":conf_mat, 'F1':f1, 'BER':ber}
            
            if len(confounder_list)>0:
                each_output['Fit_ols_deconfounding_model'] = model_fwl # NEW: old pickles will NOT have this object

            output[each_seed] = each_output
            with open(pickle_path + '.temp', 'wb') as f:
                pickle.dump(output, f)
    for key in output:
        roc_scores.append(output[key]['AUC'])
        f1_scores.append(output[key]['F1'])
        ber_scores.append(output[key]['BER'])
    print("ROC AUC mean:"+str(np.mean(roc_scores)))
    print("ROC AUC median:"+str(np.median(roc_scores)))
    print("F1 median:"+str(np.median(f1_scores)))
    print("BER median:"+str(np.median(ber_scores)))
    with open(pickle_path, 'wb') as f:
        pickle.dump(output, f)
    os.remove(pickle_path + '.temp')
    if os.path.isfile(pickle_path):
        print('Results saved successfully!')
    return(output)

### PROCESSING RESULTS ###
# get balanced error rate metric from confusion matrix
def get_BER(confusion_matrix):
        tn, fp, fn, tp = confusion_matrix.ravel()
        N = tn+fn+fp+tp
        E1 = fp/N
        E2 = fn/N
        P1 = (tn/N)+E1
        P2 = (tp/N)+E2
        BER = ((E1/P1)+(E2/P2))/2
        return BER

# return metrics given a confusion matrix
def get_pr_recall_specificity(conf_m):
    tn, fp, fn, tp = conf_m.ravel()
    pr = tp/(tp+fp)
    recall = tp/(tp+fn)
    spec = tn/(tn+fp)
    return({"pr":pr,"recall":recall,"spec":spec})

# loops over pickles in directory to extract metrics
def pickle_to_df (input_dir, export_file, tidy_file):
    df = pd.DataFrame(columns=['name', 'auc_median', 'f1_median', 'ber_median', 'pr_median', 'recall_median', 'spec_median', 'prcauc_median', 'prevalence_median'])
    df_expanded = pd.DataFrame(columns=['method', 'seed', 'auc', 'f1', 'ber', 'pr', 'recall', 'spec', 'prcauc', 'prevalence'])
    for file in os.listdir(input_dir):
        filename = os.fsdecode(file)

        pickle = pd.read_pickle(os.path.join(input_dir, file))
        print(f'pickle read: {file}')

        AUCs = []
        F1s = []
        BERs = []
        PRs = []
        recalls = []
        specs = []
        prcaucs = []
        prevalences = []

        for key in pickle.keys():
            rocauc = pickle[key]['AUC']
            f1 = pickle[key]['F1']
            ber = pickle[key]['BER']
            calced_metrics = get_pr_recall_specificity(pickle[key]['Confusion_matrix'])
            pr = calced_metrics['pr']
            recall = calced_metrics['recall']
            spec = calced_metrics['spec']
            AUCs.append(rocauc)
            F1s.append(f1)
            BERs.append(ber)
            PRs.append(pr)
            recalls.append(recall)
            specs.append(spec)

            if 'y_testset' in pickle[key].keys():
                precision_vec, recall_vec, _ = precision_recall_curve(pickle[key]['y_testset'], pickle[key]['Fit_model'].predict_proba(pickle[key]['X_testset'])[:, 1])
                prc_auc = auc(recall_vec, precision_vec)
                prcaucs.append(prc_auc)
                prev = (pickle[key]['y_testset'] == 1).sum()/len(pickle[key]['y_testset'])
                prevalences.append(prev)
                df_expanded.loc[-1] = [file, key, rocauc,f1,ber,pr,recall,spec,prc_auc, prev]
            else:
                df = df[['name', 'auc_median', 'f1_median', 'ber_median', 'pr_median', 'recall_median', 'spec_median']]
                df_expanded = df_expanded[['method', 'seed', 'auc', 'f1', 'ber', 'pr', 'recall', 'spec']]

                df_expanded.loc[-1] = [file, key, rocauc,f1,ber,pr,recall,spec]

            df_expanded.index = df_expanded.index + 1  # shifting index
            df_expanded = df_expanded.sort_index()  # sorting by index

        auc_median = np.nanmedian(AUCs)
        f1_median = np.nanmedian(F1s)
        ber_median = np.nanmedian(BERs)
        pr_median = np.nanmedian(PRs)
        recall_median = np.nanmedian(recalls)
        spec_median = np.nanmedian(specs)
        
        if 'y_testset' in pickle[key].keys():
            prcauc_median = np.nanmedian(prcaucs)
            prevalence_median = np.nanmedian(prevalences)
            df.loc[-1] = [file, auc_median,f1_median,ber_median,pr_median,recall_median,spec_median,prcauc_median,prevalence_median]
        else:
            df.loc[-1] = [file, auc_median,f1_median,ber_median,pr_median,recall_median,spec_median]
        
        df.index = df.index + 1  # shifting index
        df = df.sort_index()  # sorting by index
    df.to_csv(export_file)
    print(f'file written: {export_file}')
    df_expanded.to_csv(tidy_file)
    print(f'file written: {tidy_file}')

# process pickle to get max and mean shap value dfs
# combines pickle_to_shap_max and pickle_to_shap_mean
def get_top_feats(input_pickle):
    feature_df_list_max=[]
    feature_df_list_mean=[]
    seedlist = input_pickle.keys()
    for each_seed in seedlist:
        X_testset=input_pickle[each_seed]['X_testset']
        X_trainset=input_pickle[each_seed]['X_trainset']
        fit_model=input_pickle[each_seed]['Fit_model']
        explainer=shap.TreeExplainer(fit_model)
        explanation=explainer(X_testset)
        shap_values=explanation.values
        # shap.plots.beeswarm(explanation, max_display=50)
        # shap.plots.beeswarm(explanation, max_display=50, order=explanation.abs.max(0))
        # df_feature_max_shap = pd.DataFrame({'Feature': X_testset.columns, 'Max_Abs_SHAP_'+str(each_seed): (shap_values).max(0)})
        df_feature_max_abs_shap = pd.DataFrame({'Feature': X_testset.columns, 'Max_Abs_SHAP_'+str(each_seed): np.abs(shap_values).max(0)})
        df_feature_mean_abs_shap = pd.DataFrame({'Feature': X_testset.columns, 'Max_Abs_SHAP_'+str(each_seed): np.abs(shap_values).mean(0)})
        df_feature_max_abs_shap_sorted = df_feature_max_abs_shap.sort_values(by='Max_Abs_SHAP_'+str(each_seed), ascending=False)
        df_feature_mean_abs_shap_sorted = df_feature_mean_abs_shap.sort_values(by='Max_Abs_SHAP_'+str(each_seed), ascending=False)
        feature_df_list_max.append(df_feature_max_abs_shap_sorted)
        feature_df_list_mean.append(df_feature_mean_abs_shap_sorted)
    feature_df_full_max = feature_df_list_max[0]
    feature_df_full_mean = feature_df_list_mean[0]
    for i in range(1,len(feature_df_list_max)):
        feature_df_full_max = feature_df_full_max.merge(feature_df_list_max[i],left_on="Feature", right_on="Feature")
    for i in range(1,len(feature_df_list_mean)):
        feature_df_full_mean = feature_df_full_mean.merge(feature_df_list_mean[i],left_on="Feature", right_on="Feature")
    feature_df_full_max['Max_Max_Abs']= feature_df_full_max.iloc[:, 1:11].max(1)
    feature_df_full_max['Timepoint']=feature_df_full_max['Feature'].str.replace('_rLC.*', '', regex=True)
    feature_df_full_max['Feature']=feature_df_full_max['Feature'].str.replace('^TP[0-9]_', '', regex=True)
    feature_df_full_max.sort_values(by='Max_Max_Abs', ascending=False).head(50)

    feature_df_full_mean['Max_Max_Abs']=feature_df_full_mean.iloc[:, 1:11].max(1)
    feature_df_full_mean['Timepoint']=feature_df_full_mean['Feature'].str.replace('_rLC.*', '', regex=True)
    feature_df_full_mean['Feature']=feature_df_full_mean['Feature'].str.replace('^TP[0-9]_', '', regex=True)
    feature_df_full_mean.sort_values(by='Max_Max_Abs', ascending=False).head(50)

    feature_df_full_max=feature_df_full_max.sort_values(by='Max_Max_Abs', ascending=False).drop_duplicates(subset='Feature')
    feature_df_full_mean=feature_df_full_mean.sort_values(by='Max_Max_Abs', ascending=False).drop_duplicates(subset='Feature')

    return([feature_df_full_max, feature_df_full_mean])

# process pickle to get max and mean shap value dfs
# combines pickle_to_shap_max and pickle_to_shap_mean
def get_top_feats_with_neg(input_pickle):
    feature_df_list_max=[]
    feature_df_list_mean=[]
    seedlist = input_pickle.keys()
    for each_seed in seedlist:
        X_testset=input_pickle[each_seed]['X_testset']
        X_trainset=input_pickle[each_seed]['X_trainset']
        fit_model=input_pickle[each_seed]['Fit_model']
        explainer=shap.TreeExplainer(fit_model)
        explanation=explainer(X_testset)
        shap_values=explanation.values
        abs_shap_values = np.abs(shap_values)
        max_abs_indices = abs_shap_values.argmax(axis=0)
        # Retrieve the original SHAP values corresponding to these indices
        max_shap_values_with_sign = shap_values[max_abs_indices, range(shap_values.shape[1])]
        df_feature_max_abs_shap = pd.DataFrame({'Feature': X_testset.columns, 'Max_SHAP_'+str(each_seed): max_shap_values_with_sign})
        df_feature_mean_abs_shap = pd.DataFrame({'Feature': X_testset.columns, 'Max_SHAP_'+str(each_seed): shap_values.mean(0)})
        df_feature_max_abs_shap_sorted = df_feature_max_abs_shap.sort_values(by='Max_SHAP_'+str(each_seed), ascending=False)
        df_feature_mean_abs_shap_sorted = df_feature_mean_abs_shap.sort_values(by='Max_SHAP_'+str(each_seed), ascending=False)
        feature_df_list_max.append(df_feature_max_abs_shap_sorted)
        feature_df_list_mean.append(df_feature_mean_abs_shap_sorted)
    feature_df_full_max = feature_df_list_max[0]
    feature_df_full_mean = feature_df_list_mean[0]
    for i in range(1,len(feature_df_list_max)):
        feature_df_full_max = feature_df_full_max.merge(feature_df_list_max[i],left_on="Feature", right_on="Feature")
    for i in range(1,len(feature_df_list_mean)):
        feature_df_full_mean = feature_df_full_mean.merge(feature_df_list_mean[i],left_on="Feature", right_on="Feature")
    feature_df_full_max['Max_Max_Abs']= np.abs(feature_df_full_max.iloc[:, 1:11]).max(1)
    feature_df_full_max['Timepoint']=feature_df_full_max['Feature'].str.replace('_rLC.*', '', regex=True)
    feature_df_full_max['Feature']=feature_df_full_max['Feature'].str.replace('^TP[0-9]_', '', regex=True)
    #feature_df_full_max.sort_values(by='Max_Max_Abs', ascending=False).head(50)

    feature_df_full_mean['Max_Max_Abs']=np.abs(feature_df_full_mean.iloc[:, 1:11]).max(1)
    feature_df_full_mean['Timepoint']=feature_df_full_mean['Feature'].str.replace('_rLC.*', '', regex=True)
    feature_df_full_mean['Feature']=feature_df_full_mean['Feature'].str.replace('^TP[0-9]_', '', regex=True)
    #feature_df_full_mean.sort_values(by='Max_Max_Abs', ascending=False).head(50)

    feature_df_full_max=feature_df_full_max.sort_values(by='Max_Max_Abs', ascending=False).drop_duplicates(subset='Feature')
    feature_df_full_mean=feature_df_full_mean.sort_values(by='Max_Max_Abs', ascending=False).drop_duplicates(subset='Feature')

    return([feature_df_full_max, feature_df_full_mean])

# process pickle to get correlations of shap with the feature
def get_shap_feat_correlations(input_pickle):
    seedlist = input_pickle.keys()
    corrs = {}
    p_val = 0.05
    for each_seed in seedlist:
        corr = []
        X_testset=input_pickle[each_seed]['X_testset']
        fit_model=input_pickle[each_seed]['Fit_model']
        explainer=shap.TreeExplainer(fit_model)
        explanation=explainer(X_testset)
        shap_values=explanation.values
        shap.plots.beeswarm(explanation, max_display=50)
        shap.plots.beeswarm(explanation, max_display=50, order=explanation.abs.max(0))
        df_feature_max_shap = pd.DataFrame({'Feature': X_testset.columns, 'Max_Abs_SHAP_'+str(each_seed): (shap_values).max(0)})
        for i in range(len(np.transpose(shap_values))):
            zipped = pd.DataFrame(data={'shap':np.transpose(shap_values)[i],'feat':X_testset.iloc[:,i]})
            zipped = zipped.dropna()
            pearson = stats.pearsonr(zipped['shap'],zipped['feat'])
            if pearson.pvalue<=(p_val/len(X_testset.columns)):
                corr.append(np.sign(pearson.statistic))
            else:
                corr.append(np.nan)
        corrs[each_seed] = corr
    df = pd.DataFrame(data=corrs, index= X_testset.columns)
    return(df)

# rank features in terms of shap, and pull out features above a certain cutoff for a certain number of seeds
def rank_and_filter(df, seed_cutoff, rank_cutoff):
    # assumes first col is feature name and last two cols are Max_Max_Abs and Timepoint, hence iloc[:, 1:len(df.columns)-2]
    df.replace(0, np.nan, inplace=True)
    df_rank = df.assign(**df.iloc[:, 1:len(df.columns)-2].rank(axis = 0, ascending = False, na_option='keep').astype('Int64'))
    df_rank['seeds_above_cutoff'] = df_rank.iloc[:, 1:len(df.columns)-2].apply(lambda row : (row <= rank_cutoff).sum(), axis=1)
    df_rank_filt=df_rank[df_rank['seeds_above_cutoff']>=seed_cutoff]
    print(df_rank_filt.shape)
    return(df_rank_filt)

### Plotting data
def plot_ROC_AUC(pickle, type='mean', plot_title=None) :
    # Plots individual and mean or median AUC plots from a single pickle
    # Uses mean or median to calculate AUC from each seed in pickle
    # from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    # Extracting results per replicate
    for i in pickle.keys():  
        pickle_rep=pickle[i]

        # Get each ROC curve
        viz = RocCurveDisplay.from_estimator(
            pickle_rep['Fit_model'],
            pickle_rep['X_testset'],
            pickle_rep['y_testset'],
            name="ROC fold {}".format(i),  # Label for the ROC curve of this fold
            alpha=0.3,  # Transparency of the ROC curve
            lw=1,  # Line width of the ROC curve
            ax=ax  # Axis to plot the ROC curve on
        )

        # Interpolate (deduce) the true positive rates (TPR) at the mean false positive rate (FPR)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        
        # Append the interpolated TPR values and AUC (Area Under Curve) for this fold
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    if type=="mean":
        mean_tpr = np.mean(tprs, axis=0)
    elif type=="median":
        mean_tpr = np.median(tprs, axis=0)
    else:
        print("incorrect type specified, options are 'mean' and 'median'")
        
    mean_tpr[-1] = 1.0
    smoothed_mean_tpr=savgol_filter(mean_tpr, window_length=11, polyorder=2)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        smoothed_mean_tpr,
        color="b",
        label=f"{type} ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    smoothed_std_tpr=savgol_filter(std_tpr, window_length=11, polyorder=2)
    tprs_upper = np.minimum(smoothed_mean_tpr + smoothed_std_tpr, 1)
    tprs_lower = np.maximum(smoothed_mean_tpr - smoothed_std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=plot_title,
    )
    ax.legend(loc="lower right")
    plt.show()

def compare_plot_ROC_AUC(pickles, plot_names, colorset=['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#F0E442'], type='mean',
                         plot_title=None):
    # Plots the mean or median AUC plots from multiple ML runs together
    # Requires a list of pickles, and a list of labels for the plot.
    # Colorset is customizable as 'colorset'
    # Uses mean or median to calculate AUC from each seed in pickle
    # adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    n=0
    fig, ax = plt.subplots()
    # Extracting results per replicate
    for pickle in pickles:
        tprs = []
        aucs = []
        for i in pickle.keys():  
            pickle_rep=pickle[i]
            mean_fpr = np.linspace(0, 1, 100)
            # Get each ROC curve
            viz = RocCurveDisplay.from_estimator(
                pickle_rep['Fit_model'],
                pickle_rep['X_testset'],
                pickle_rep['y_testset'],
                label=None,
                alpha=0,  # Transparency of the ROC curve
                lw=0,  # Line width of the ROC curve
                ax=ax  # Axis to plot the ROC curve on
            )

            # Interpolate (deduce) the true positive rates (TPR) at the mean false positive rate (FPR)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            
            # Append the interpolated TPR values and AUC (Area Under Curve) for this fold
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        if type=="mean":
            mean_tpr = np.mean(tprs, axis=0)
        elif type=="median":
            mean_tpr = np.median(tprs, axis=0)
        else:
            print("incorrect type specified, options are 'mean' and 'median'")
            
        mean_tpr[-1] = 1.0
        smoothed_mean_tpr=savgol_filter(mean_tpr, window_length=11, polyorder=2)
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            smoothed_mean_tpr,
            color=colorset[n],
            label=f"{plot_names[n]} (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        smoothed_std_tpr=savgol_filter(std_tpr, window_length=11, polyorder=2)
        tprs_upper = np.minimum(smoothed_mean_tpr + smoothed_std_tpr, 1)
        tprs_lower = np.maximum(smoothed_mean_tpr - smoothed_std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=colorset[n],
            alpha=0.1,
            #label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=plot_title,
        )
        ax.legend(loc="lower right")
        n=n+1
    return fig

def compare_plot_PR(pickles, plot_names, colorset=['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#F0E442'], type='mean',
                         plot_title=None, label=None, ci_type='std'):
    # Plots the mean or median precision-recall plots from multiple ML runs together
    # Requires a list of pickles, and a list of labels for the plot.
    # Colorset is customizable as 'colorset'
    # Uses mean or median to generate a PR curve from each seed in pickle
    # Intended to be used for comparisons within the same sample set, otherwise the random baseline will differ
    n=0
    fig, ax = plt.subplots()
    # Extracting results per replicate
    for pickle in pickles:
        Prec = [] # y-axis (Precision)
        AP = [] # reported value
        val = [] # random baseline
        for i in pickle.keys():  
            pickle_rep=pickle[i]
            mean_recall = np.linspace(0, 1, 36) # x-axis (Recall)
            # Get each PR curve
            viz = PrecisionRecallDisplay.from_estimator(
                pickle_rep['Fit_model'],
                pickle_rep['X_testset'],
                pickle_rep['y_testset'],
                label=None,
                alpha=0,  # Transparency of the PR curve
                lw=0,  # Line width of the PR curve
                ax=ax  # Axis to plot the PR curve on
            )

            # Interpolate (deduce) the precision at the mean recall
            interp_Prec = np.interp(mean_recall, viz.recall[::-1], viz.precision[::-1])
            #interp_Prec[0] = 1

            # Append the interpolated TPR values and AUC (Area Under Curve) for this fold
            Prec.append(interp_Prec)
            AP.append(viz.average_precision)

        # Calculate random level
        val.append(np.mean(pickle_rep['y_testset'] == 1))

        # Calculate mean/median precision 
        if type=="mean":
            mean_prec = np.mean(Prec, axis=0)
            mean_AP = np.mean(AP)
        elif type=="median":
            mean_prec = np.median(Prec, axis=0)
            mean_AP = np.median(AP)
        else:
            print("incorrect type specified, options are 'mean' and 'median'")

        precision_array = np.array(Prec)
        mean_precision = np.mean(precision_array, axis=0)

        # Calculate std/95ci
        if ci_type == 'std':
            error = np.std(precision_array, axis=0)
            label = '±1 std. dev.'
        elif ci_type == '95ci':
           error = sem(precision_array, axis=0) * 1.96
           label = '95% CI'
        else:
            raise ValueError("ci_type must be 'std' or '95ci'")

        upper = np.minimum(mean_precision + error, 1)
        lower = np.maximum(mean_precision - error, 0)

        std_auc = np.std(AP)

        ax.plot(mean_recall, mean_precision, color=colorset[n],
                label=f'{plot_names[n]} (AP = {mean_AP:.2f} ± {std_auc:.2f})', lw=2)
        ax.fill_between(mean_recall, lower, upper, color=colorset[n], alpha=0.2, label=None)   
        
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=plot_title,
        )
        ax.legend(loc="lower right")

        n=n+1
    
    if len(set(val)) > 1:
        print("Warning, more than one random baseline value detected!")
    else: 
        plt.axhline(val[0], linestyle='--', color='grey', label=f"Random ({val[0]:.2f})")
        plt.legend()

    return fig
