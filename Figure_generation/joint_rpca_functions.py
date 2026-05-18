
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import networkx as nx
import pandas as pd
from statannotations.Annotator import Annotator
from scipy import stats

def ord_scatter_3D(ord_dict, hue, hue_order, legend_title="",
                   pcs=["PC1","PC2","PC3"], palette="tab10", ncol=1,
                   subplots=(2,3), figsize=(25, 10), elev=20, axim=120,
                   save_fig=False, save_path=None,):

    if subplots is not None:
        fig, axn = plt.subplots(subplots[0], subplots[1], figsize=figsize,
                                subplot_kw=dict(projection='3d'))
        axn = axn.flatten()
    else:
        fig = plt.figure(figsize=figsize).add_subplot(projection='3d')
        axn = [fig]

    for ax, (tblid, ord_plt) in zip(axn, ord_dict.items()):
        #assign colors
        if hue_order is None:
            c = ord_plt[hue]
            cmap = palette
            patches = []
        else:
            cmap = None
            color_dict = dict(zip(hue_order, palette))
            ord_plt['Color'] = ord_plt[hue].map(color_dict)
            ord_plt = ord_plt.dropna(subset=['Color'])
            c = ord_plt['Color']
            patches = [mpatches.Patch(color=value, label=key) for 
                       key, value in color_dict.items()]
            
        ax.scatter(xs=ord_plt[pcs[0]], ys=ord_plt[pcs[1]], s=25, zs=ord_plt[pcs[2]],
                   c=c, marker='o', cmap=cmap)
        ax.set_xlabel("{}".format(pcs[0]), color='black', weight='bold', fontsize=16)
        ax.set_ylabel("{}".format(pcs[1]), color='black', weight='bold', fontsize=16)
        ax.set_zlabel("{}".format(pcs[2]), color='black', weight='bold', fontsize=16)
        ax.set_title(tblid, color='black', weight='bold', fontsize=20, y=1)

        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 15

        # fix backround
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_facecolor('white')
        ax.set_axisbelow(True)
        for child in ax.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color('grey')
        ax.grid(True)
        ax.view_init(elev=elev, azim=axim)

    if subplots is not None:
        legend = ax.legend(handles=patches, loc=9, title='', prop={'size':16},
                           bbox_to_anchor=(-0.7, 2.5), fancybox=True, framealpha=.0,
                           ncol=ncol, markerscale=3.5)
        legend.get_title().set_fontsize('16')
        # increase the line width in the legend 
        for line in legend.get_lines()[:]:
            line.set_linewidth(2.0)
        for line in legend.get_lines()[:]:
            line.set_linewidth(2.0)
        fig.subplots_adjust(wspace=-0.5, hspace=0.2)
    
    else:
        legend = ax.legend(handles=patches, loc=2, bbox_to_anchor=(0.7, 1),
                           title=legend_title, prop={'size':16}, fancybox=True, 
                           framealpha=.0, ncol=ncol, markerscale=3.5)
        legend.get_title().set_fontsize('18')

        # increase the line width in the legend 
        for line in legend.get_lines()[:]:
            line.set_linewidth(2.0)
        for line in legend.get_lines()[:]:
            line.set_linewidth(2.0)

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_path, dpi=600, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
    
    plt.show()

#ref: https://bbquercus.medium.com/adding-statistical-significance-asterisks-to-seaborn-plots-9c8317383235
def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    elif pvalue > 0.05 and pvalue < 0.1:
        return "~"
    return "ns"

plot_title = {'micro': 'Micronutrients', 'macro': 'Macronutrients', 
              'biocrates': 'Targeted LC-MS', 'hmo': 'HMOs',
              'untargeted_all': 'Untargeted (all)', 
              'untargeted_sapient': 'Untargeted LC-MS'}

def find_log_ratios(tables_dict, feaure_tables_dict, nfeatures=5,
                     nhmos=5, axis_use='PC1', quantile=0.9):

    logratios_all = {}
    quantile = 0.90

    for modality in feaure_tables_dict.keys():
        ranks_ = feaure_tables_dict[modality]
        table_ = tables_dict[modality].T
        if (modality=='hmo'):
            num = ranks_.index[:nhmos]
            den = ranks_.index[-nhmos:]
            print("HMO num: ", num.tolist())
            print("HMO den: ", den.tolist())
        elif modality=='micro':
            # get top and bottom features
            num = ranks_.index[:nfeatures]
            den = ranks_.index[-nfeatures:]
            print("Micro num: ", num.tolist())
            print("Micro num: ", den.tolist())
        elif modality=='macro':
            num = ranks_[ranks_[axis_use] > 0].index
            den = ranks_[ranks_[axis_use] < 0].index
            print("Macro num: ", num.tolist())
            print("Macro den: ", den.tolist())
        else:
            # get top and bottom features
            top_q = ranks_[axis_use].quantile(quantile)
            bottom_q = ranks_[axis_use].quantile(1-quantile)
            print(modality, "Top quantile: ", np.round(top_q, 2))
            print(modality, "Bottom quantile: ", np.round(bottom_q, 2))
            num = ranks_[ranks_[axis_use] >= top_q].index
            den = ranks_[ranks_[axis_use] <= bottom_q].index
            print("Numerator:", num.shape)
            print("Denominator:", den.shape)
        print()
        lr_ = np.log(table_.loc[num, :].sum(0)) - np.log(table_.loc[den, :].sum(0))
        lr_[~np.isfinite(lr_)] = np.nan
        logratios_all[modality] = lr_

    return logratios_all

def plot_log_ratios(log_ratio_dict, metadata, phenotype='BEP', timepoint='Timepoint',
                    modality_lst = ['micro', 'biocrates', 'untargeted_all'], palette = None,
                    legend_labels = None, legend_title = None, legend_loc='lower right',
                    hue_order = None, plot_subtitle = 'Log-ratio Trajectories', axis_use='',
                    shared_set = False, savefig = False, save_path = None, make_legend = False,
                    **kwargs):
    
    #prepare df for plotting
    logratios_df = pd.DataFrame(log_ratio_dict)
    logratios_df = logratios_df.join(metadata)
    phenotype_vals = logratios_df[phenotype].unique()
    #remove nan values
    phenotype_vals = [x for x in phenotype_vals if str(x) != 'nan']
    timepoint_vals = logratios_df[timepoint].unique()
    #sort timepoints in ascending order
    timepoint_vals.sort()
    #remove nan values
    timepoint_vals = [x for x in timepoint_vals if str(x) != 'nan']

    #boxplot
    fig, ax = plt.subplots(1, len(modality_lst), figsize=(3.5*len(modality_lst), 4.25))

    if shared_set:
        y_label = fr'$\mathrm{{log\left(\frac{{Shared\ {axis_use}\ top\ features}}{{Shared\ {axis_use}\ bottom\ features}}\right)}}$'
    else:
        y_label = fr'$\mathrm{{log\left(\frac{{{axis_use}\ top\ features}}{{{axis_use}\ bottom\ features}}\right)}}$'

    for i, modality in enumerate(modality_lst):
        print(modality)
        sns.pointplot(x=timepoint, y=modality, hue=phenotype, hue_order=hue_order,
                      palette=palette, data=logratios_df, ax=ax[i], **kwargs)
        #run Mann-Whitney U test
        pvalues = []
        for x in timepoint_vals:
            stat, pval = stats.mannwhitneyu(
                logratios_df[(logratios_df[timepoint] == x) & (logratios_df[phenotype] == phenotype_vals[0])][modality],
                logratios_df[(logratios_df[timepoint] == x) & (logratios_df[phenotype] == phenotype_vals[1])][modality],
                nan_policy='omit')
            #round p-value
            print("Timepoint:", x, "pvalue:", np.round(pval,4))
            pvalues.append(pval)
        
        ymin, ymax = ax[i].get_ylim()
        yrange = ymax - ymin
        y_position = ymax - 0.05 * yrange

        for idx, pval in zip(timepoint_vals, pvalues):
            p_format = convert_pvalue_to_asterisks(pval)
            if p_format == "ns":
                #round to 2 decimal places
                p_format = f'{pval:.2f}'
            ax[i].text(x=idx-1, y=y_position, s=p_format,
                    ha='center', va='bottom', fontsize=10)

        #set labels
        ax[i].set_title(plot_title[modality], y=1.05, fontsize=14, weight='bold')
        ax[i].set_ylabel(y_label, color='black', weight='bold', fontsize=12)
        ax[i].set_xlabel("Timepoint", color='black', weight='bold', fontsize=12)
        #set legend to bottom right corner
        if make_legend:
            handles, labels = ax[i].get_legend_handles_labels()
            if legend_labels is not None:
                labels = [legend_labels[str(x)] for x in labels]
            if legend_title is None:
                legend_title = phenotype
            ax[i].legend(handles, labels, title=legend_title, loc=legend_loc)
        else:
            ax[i].get_legend().remove()
        
        #remove top and right spines
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        #ingrease tick font size
        ax[i].tick_params(axis='both', which='major', labelsize=11)
        
    plt.suptitle(plot_subtitle, y=0.95,
                 color='black', weight='bold', fontsize=14)
    plt.tight_layout()

    if savefig:
        plt.savefig(save_path, dpi=1000, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
    
    plt.show()

def subject_boxplot(plot_df, x_, y_, order_, palette_, prop_explained_,
                    xlabel_, title_, save_fig=False, save_path=None,
                    figsize_= (3.5, 4.5)):
    
    fig = plt.figure(figsize=figsize_)
    ax = fig.gca()

    sns.boxplot(x=x_, y=y_, data=plot_df, order=order_, zorder=1, showfliers=False,
                palette=palette_, ax=ax)
    #add points
    sns.stripplot(x=x_, y=y_, data=plot_df, order=order_, linewidth=0.2, edgecolor='black', zorder=2,
                  alpha=0.5, jitter=0.2, palette=palette_, ax=ax)
    sns.boxplot(x=x_, y=y_, data=plot_df, order=order_, zorder=3, showfliers=False,
                palette=palette_, ax=ax, boxprops=dict(facecolor='none'))
    #add significance
    annotator = Annotator(ax, [(order_[0], order_[1])], x=x_, y=y_, data=plot_df)
    annotator.configure(test="Mann-Whitney", text_format='star', verbose=1).apply_and_annotate()

    ax.set_xlabel(xlabel_, color='black', weight='bold', fontsize=14)
    ax.set_ylabel("{} ({}%)".format(y_, prop_explained_[y_]), 
                  color='black', weight='bold', fontsize=14)
    
    plt.title(title_, color='black', weight='bold', fontsize=14)

    ax.set_facecolor('white')
    ax.set_axisbelow(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', 
                    facecolor=fig.get_facecolor(), edgecolor='none')
    
    plt.show()

def extract_feature_loadings(ord_results, raw_tables, axis_use='PC1',
                             modalities=['micro', 'hmo', 
                                         'biocrates', 'untargeted_sapient']):
    
    #extract misame feature loadings
    ord_feats = ord_results.features.copy()
    ord_feats.rename(columns={0:'PC1', 1:'PC2', 2:'PC3'}, inplace=True)

    feature_tables = {}

    for id in modalities:
        table_df = raw_tables[id]
        id_set = set(ord_feats.index) & set(table_df.columns)
        ord_df_ = ord_feats.loc[list(id_set), :].copy()
        ord_df_ = ord_df_.sort_values(by=axis_use, ascending=False)
        feature_tables[id] = ord_df_

    return feature_tables

def network(links_df, feature_cor, feature_colors,
            color_map, node_size=800, ncol=4,
            num_iterations=5, figsize=(19, 14),
            save_fig=False, save_path=None):
    
    for iteration in range(num_iterations):

        # Build your graph
        G = nx.from_pandas_edgelist(links_df, 'var1', 'var2')
    
        # Plot the network:
        pos = nx.spring_layout(G, k=1.8, seed=iteration)
        fig, ax = plt.subplots(1, 1, figsize=figsize) 
        y_off = 3

        nx.draw(G, with_labels=True,
                pos = {k:([v[0], v[1]+y_off]) for k,v in pos.items()},
                
                node_color=[feature_colors[node[0]]
                            for node in G.nodes(data=True)], 
                #node_size=[assoc_map_label[node[0]]*4
                #            for node in G.nodes(data=True)], 
                node_size=node_size,
                edge_color=[(feature_cor.loc[u, v]) for u, v in G.edges],
                edge_cmap= plt.cm.Blues,
                linewidths=5,
                width=1,
                font_size=16, font_weight='bold', ax=ax)

        handles_ = [mpatches.Patch(color=c_, label=modality_) for modality_, c_ in color_map.items()]
        legend = ax.legend(handles=handles_, loc='upper center', bbox_to_anchor=(0.5, 0.05),
                            prop={'size':20}, title="", fancybox=True,
                            framealpha=.0, ncol=ncol, markerscale=1.5)
        #legend.get_title().set_fontsize('20')
        # increase the line width in the legend 
        for line in legend.get_lines()[:]:
            line.set_linewidth(2.0)
        for line in legend.get_lines()[:]:
            line.set_linewidth(2.0)

        plt.tight_layout()

        if save_fig:
            plt.savefig(save_path, dpi=600, 
                        bbox_inches='tight',
                        facecolor=fig.get_facecolor(), 
                        edgecolor='none')
    
        plt.show()

def shared_log_ratios(tables_dict, num_dict, denom_dict):

    logratios_all = {}

    for modality in tables_dict.keys():    

        table_ = tables_dict[modality].T
        num = num_dict[modality]
        den = denom_dict[modality]
        lr_ = np.log(table_.loc[num, :].sum(0)) - np.log(table_.loc[den, :].sum(0))
        lr_[~np.isfinite(lr_)] = np.nan
        logratios_all[modality] = lr_

    return logratios_all