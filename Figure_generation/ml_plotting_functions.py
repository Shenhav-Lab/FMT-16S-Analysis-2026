# Data importing
import numpy as np  # Fundamental library for numerical computations
import pickle  # For loading serialized Python objects (pickles)
from sklearn.metrics import auc  # For calculating Area Under the Curve
from sklearn.metrics import RocCurveDisplay  # To plot individual ROC curves
from scipy.signal import savgol_filter  # For smoothing curves
import matplotlib.pyplot as plt  # Primary plotting library
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score

### Functions for Plotting data
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
        mean_auc = np.mean(aucs)
    elif type=="median":
        mean_tpr = np.median(tprs, axis=0)
        mean_auc = np.median(aucs)
    else:
        print("incorrect type specified, options are 'mean' and 'median'")
        
    mean_tpr[-1] = 1.0
    smoothed_mean_tpr=savgol_filter(mean_tpr, window_length=11, polyorder=2)
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
            mean_auc = np.mean(aucs)
        elif type=="median":
            mean_tpr = np.median(tprs, axis=0)
            mean_auc = np.median(aucs)
        else:
            print("incorrect type specified, options are 'mean' and 'median'")
            
        mean_tpr[-1] = 1.0
        smoothed_mean_tpr=savgol_filter(mean_tpr, window_length=11, polyorder=2)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            smoothed_mean_tpr,
            color=colorset[n],
            label=f"{plot_names[n]} (AUC=%0.2f$\pm$%0.2f)" % (mean_auc, std_auc),
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
                label=f'{plot_names[n]} (AP={mean_AP:.2f}±{std_auc:.2f})', lw=2)
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
