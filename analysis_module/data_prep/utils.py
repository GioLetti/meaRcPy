#%%
import pickle
import os
#import yaml
import matplotlib.pyplot as plt
from typing import Union
from matplotlib.colors import from_levels_and_colors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd
import re
import seaborn as sns
from statannotations.Annotator import Annotator
import copy
import statsmodels.api as sm
from scipy.stats import ttest_ind,linregress,kruskal
from matplotlib.ticker import MultipleLocator,AutoMinorLocator,FixedLocator
from scipy.io import savemat


def adjust_font_dimension(legend_size=10,title_size=15,label_size=12,x_ticks=10,y_ticks=10):

    plt.rc('legend',fontsize=legend_size) # using a size in points
    plt.rc('axes',titlesize=title_size) 
    
    plt.rc('axes',labelsize=label_size) # using a size in points
    plt.rc('xtick',labelsize=x_ticks) 
    plt.rc('ytick',labelsize=y_ticks) 


def check_if_file_exist(path_to_check,message):
    '''
    Check existence of a file
    '''
    if not os.path.exists(path_to_check): 
        raise FileNotFoundError(message)


def export_cm_matrix(cm_analysis_data,m:int,a:float,output_path,ext='.npy'):
    '''
    function used for exporting the CM matrix of the specified MEArc model parametrized by a (alpha) and m parameters.
    
    Input
        - cm_analysis_data: either the path (str) to the cm_analysis_results.pickle file or the direct data
        - m: m parameter of MEArc model
        - a: alpha parameter of MEArc model
        - output_path: desired output path
        - ext:  default = '.npy' the matrix will be saved as nunpy matrix. if '.mat' the matrix will be saved as mat file using savemat function from scipy

    '''
    if isinstance(cm_analysis_data,str):
        cms = open_pickle_file(cm_analysis_data)

    else:
        cms = cm_analysis_data

    m_values = list(cms.keys())

    alpha_values = list(cms[m_values[0]].keys())

    if m not in m_values:
        raise ValueError(f'Desired m value not found. Available are: {m_values}')

    if a not in alpha_values:
        raise ValueError(f'Desired alpha value not found. Available are: {alpha_values}') 

    cm = cms[m][a]['avg_cm']

    if ext=='.npy':

        np.save(os.path.join(output_path,f'avg_cm_{m}_{a}'+ext),cm)

    elif ext=='.mat':
        savemat(os.path.join(output_path,f'avg_cm_{m}_{a}'+ext),cm)

    else:
        raise ValueError('CM matrix can be saved as either as .npy or .mat file')


    



#%%
def save_pickle_file(object, # object to save
                     path:os.path): # path where to save
    '''
    Function saves the object as pickle file
    '''

    with open(path, 'wb') as handle:
        pickle.dump(object, handle, protocol=4)


def open_pickle_file(pickel_file_path:str
                    ):
    '''
    function open a pickle file
    '''

    check_if_file_exist(pickel_file_path,f'file {pickel_file_path} does not exist')
    pkl_file = open(pickel_file_path,'rb')
    open_pickle = pickle.load(pkl_file)

    return open_pickle

def take_best_m_alpha(validationf_file_path):

    df = pd.read_csv(validationf_file_path,sep='\t')
    best_idx = df['avg_rmse'].argmin()

    alpha = df['alpha'][best_idx] # optimal alpha
    m = df['m'][best_idx] # optimal m

    return m,alpha




def plot_mea_map(unfiltered_channel_list,filtered_channel_list,channel_integral,stim_electrode,title,output_path,bin=False,bin_err = False,exp_type='',layout = (12,5)):


    if exp_type=='sim':
        mea_map = np.zeros(channel_integral.shape[0])
        mea_map[:]=np.nan

        for ch in range(channel_integral.shape[0]):

            mea_map[ch] = channel_integral[ch]

        mea_map=np.reshape(mea_map,layout)

        #!TODO adjust the map for considering filtering even for simulations
        stim_electrode = stim_electrode

        x_stim = int(stim_electrode//layout[1])
        y_stim = int(stim_electrode%layout[1])

        entry_legend_star = Line2D([0], [0], color='yellow', marker='*',markersize=30, linestyle='None', label='Stimulated')
        
        if bin==True:
            cmap, norm = from_levels_and_colors([0, 0.5, 1.1], ['tab:blue', 'r'])

            
            # create a patch (proxy artist) for every color 
            patches = [mpatches.Patch(color='tab:blue',label = 'Inactive'),mpatches.Patch(color='r',label = 'Active'),entry_legend_star]
            
            plt.close()
            fig,ax = plt.subplots()
            
            ax.scatter(x = [y_stim],y=[x_stim],marker='*',s=500,c='yellow')
            ax.imshow(mea_map,interpolation='none', cmap=cmap, norm=norm)
            ax.set_xticks(np.arange(0,5,1))
            ax.set_yticks(np.arange(0,12,1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            ax.grid(True,which='minor',color='black', linestyle='-', linewidth=1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            ax.legend(handles=patches,bbox_to_anchor=(1.7, 1))
            ax.set_title(title)
            plt.tick_params(bottom = False,left=False) 
            plt.savefig(output_path,bbox_inches='tight')
            plt.close()

            

        elif bin_err == True:
            cmap, norm = from_levels_and_colors([0, 0.5, 1.1], ['olive', 'mediumpurple'])

            
            # create a patch (proxy artist) for every color 
            patches = [mpatches.Patch(color='mediumpurple',label = 'Not predicted'),mpatches.Patch(color='olive',label = 'Predicted'),entry_legend_star]
            

            plt.close()
            fig,ax = plt.subplots()
            #ax.scatter(x=y_pos_to_annotate,y=x_pos_to_annotate,marker='x',s=1000,c='black')
            ax.scatter(x = [y_stim],y=[x_stim],marker='*',s=500,c='yellow')
            ax.imshow(mea_map,interpolation='none', cmap=cmap, norm=norm)
            ax.set_xticks(np.arange(0,5,1))
            ax.set_yticks(np.arange(0,12,1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            ax.grid(True,which='minor',color='black', linestyle='-', linewidth=1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.legend(handles=patches,bbox_to_anchor=(1.7, 1))
            ax.set_title(title)
            plt.tick_params(bottom = False,left=False) 
            plt.savefig(output_path,bbox_inches='tight')
            plt.close()
        

        else:


            plt.close()
            fig,ax = plt.subplots()
            #ax.scatter(x=y_pos_to_annotate,y=x_pos_to_annotate,marker='x',s=1000,c='black')
            ax.scatter(x = [y_stim],y=[x_stim],marker='*',s=500,c='yellow')
            im = ax.imshow(mea_map,interpolation='none', cmap='coolwarm')
            fig.colorbar(im)
            ax.set_xticks(np.arange(0,5,1))
            ax.set_yticks(np.arange(0,12,1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            ax.grid(True,which='minor',color='black', linestyle='-', linewidth=1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.legend(handles=[entry_legend_star],bbox_to_anchor=(3.9, 1))
            ax.set_title(title)
            plt.tick_params(bottom = False,left=False) 
            plt.savefig(output_path,bbox_inches='tight')
            plt.close()


    else:
        mea_map = np.zeros((8,8))
        mea_map[:]=np.nan
    
    
        for ch in range(channel_integral.shape[0]):

            ch_idx = str(filtered_channel_list[ch])
            col_idx = int(ch_idx[0])-1 # index of python
            row_idx = int(ch_idx[1])-1

            mea_map[row_idx,col_idx] = channel_integral[ch]

        

        x_pos_to_annotate = [4]
        y_pos_to_annotate = [0]

        for ch in range(len(unfiltered_channel_list)):

            ch_idx = unfiltered_channel_list[ch]

            if int(ch_idx) not in filtered_channel_list:

                col_idx = int(ch_idx[0])-1 # index of python
                row_idx = int(ch_idx[1])-1

                y_pos_to_annotate.append(col_idx)
                x_pos_to_annotate.append(row_idx)

        stim_electrode = str(stim_electrode)
        x_stim = int(stim_electrode[1])-1
        y_stim = int(stim_electrode[0])-1

        entry_legend_x = Line2D( [0], [0],color='black', marker='x',markersize=30,markeredgewidth=1,linestyle='None', label='Not analyzed')
        entry_legend_star = Line2D([0], [0], color='yellow', marker='*',markersize=30, linestyle='None', label='Stimulated')
        

        if bin==True:
            cmap, norm = from_levels_and_colors([0, 0.5, 1.1], ['tab:blue', 'r'])

            
            # create a patch (proxy artist) for every color 
            patches = [mpatches.Patch(color='tab:blue',label = 'Inactive'),mpatches.Patch(color='r',label = 'Active'),entry_legend_x,entry_legend_star]
            
            plt.close()
            fig,ax = plt.subplots()
            ax.scatter(x=y_pos_to_annotate,y=x_pos_to_annotate,marker='x',s=1000,c='black')

            ax.scatter(x = [y_stim],y=[x_stim],marker='*',s=500,c='yellow')
            ax.imshow(mea_map,interpolation='none', cmap=cmap, norm=norm)
            ax.set_xticks(np.arange(0,8,1))
            ax.set_yticks(np.arange(0,8,1))
            ax.set_xticklabels(np.arange(1,9,1))
            ax.set_yticklabels(np.arange(1,9,1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            ax.grid(True,which='minor',color='black', linestyle='-', linewidth=1)
            ax.legend(handles=patches,bbox_to_anchor=(1.25, 1))
            ax.set_title(title)
            plt.savefig(output_path,bbox_inches='tight')
            plt.close()

            

        elif bin_err == True:
            cmap, norm = from_levels_and_colors([0, 0.5, 1.1], ['olive', 'mediumpurple'])

            
            # create a patch (proxy artist) for every color 
            patches = [mpatches.Patch(color='mediumpurple',label = 'Not predicted'),mpatches.Patch(color='olive',label = 'Predicted'),entry_legend_x,entry_legend_star]
            

            plt.close()
            fig,ax = plt.subplots()
            ax.scatter(x=y_pos_to_annotate,y=x_pos_to_annotate,marker='x',s=1000,c='black')
            ax.scatter(x = [y_stim],y=[x_stim],marker='*',s=500,c='yellow')
            ax.imshow(mea_map,interpolation='none', cmap=cmap, norm=norm)
            ax.set_xticks(np.arange(0,8,1))
            ax.set_yticks(np.arange(0,8,1))
            ax.set_xticklabels(np.arange(1,9,1))
            ax.set_yticklabels(np.arange(1,9,1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            ax.grid(True,which='minor',color='black', linestyle='-', linewidth=1)
            ax.legend(handles=patches,bbox_to_anchor=(1.25, 1))
            ax.set_title(title)

            plt.savefig(output_path,bbox_inches='tight')
            plt.close()
        

        else:


            plt.close()
            fig,ax = plt.subplots()
            ax.scatter(x=y_pos_to_annotate,y=x_pos_to_annotate,marker='x',s=1000,c='black')
            ax.scatter(x = [y_stim],y=[x_stim],marker='*',s=500,c='yellow')
            im = ax.imshow(mea_map,interpolation='none', cmap='coolwarm')
            fig.colorbar(im)
            ax.set_xticks(np.arange(0,8,1))
            ax.set_yticks(np.arange(0,8,1))
            ax.set_xticklabels(np.arange(1,9,1))
            ax.set_yticklabels(np.arange(1,9,1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            ax.grid(True,which='minor',color='black', linestyle='-', linewidth=1)
            ax.legend(handles=[entry_legend_x,entry_legend_star],bbox_to_anchor=(2.4, 1))
            ax.set_title(title)
            plt.savefig(output_path,bbox_inches='tight')
            plt.close()

        

def fix_roc(binarized_gt_tmp,amplitude_psth,single_pred_path,gt_thr):
     
    if (np.sum(binarized_gt_tmp)/len(binarized_gt_tmp)) == 1:
        
        binarized_gt_tmp = np.append(binarized_gt_tmp,0) # to avoid zero division in roc
        amplitude_psth = np.append(amplitude_psth,0)
        roc_curve_path = os.path.join(single_pred_path,'Response_ROC(.pdf')

    elif (np.sum(binarized_gt_tmp)/len(binarized_gt_tmp)) == 0:

        binarized_gt_tmp = np.append(binarized_gt_tmp,1) # to avoid zero division in roc
        amplitude_psth = np.append(amplitude_psth,1)
        roc_curve_path = os.path.join(single_pred_path,'Response_ROC(.pdf')
    else:

        roc_curve_path = os.path.join(single_pred_path,'Response_ROC.pdf')

    return binarized_gt_tmp,amplitude_psth,roc_curve_path




def find_best_intensity_auc_xrmse(auc_list,xrmse_list,intensity_values):

    # cc_arr = np.array(cc_list)
    # cc_arr = np.where(np.isnan(cc_arr),-2,cc_arr)
    xrmse_arr = np.array(xrmse_list)
    int_arr = np.array(intensity_values)

    

    # idx_of_maximum_cc = np.where(cc_arr == np.max(cc_arr))[0]

    
    # min_xrmse_to_idx = xrmse_arr[idx_of_maximum_cc[0]]

    # idx_to_use = idx_of_maximum_cc[0]

    # for idx in idx_of_maximum_cc[1:]:
    #     if xrmse_arr[idx]<min_xrmse_to_idx:
    #         min_xrmse_to_idx = xrmse_arr[idx]
    #         idx_to_use = idx

    # return int_arr[idx_to_use]


    auc_arr = np.array(auc_list)

    idx_of_minimum_xrmse = np.where(xrmse_arr == np.min(xrmse_arr))[0]

    max_auc_at_idx = auc_arr[idx_of_minimum_xrmse[0]]

    idx_to_use = idx_of_minimum_xrmse[0]

    if len(idx_of_minimum_xrmse)>1:
        

        for idx in idx_of_minimum_xrmse[1:]:
         if auc_arr[idx]<max_auc_at_idx:
             max_auc_at_idx = auc_arr[idx]
             idx_to_use = idx

    return int_arr[idx_to_use]


def find_best_threshold_response(fpr,tpr,ths):

    fpr = np.array(fpr)
    tpr = np.array(tpr)
    ths = np.array(ths)

    idx_of_maximum_tpr = np.where(tpr == np.max(tpr[tpr<1]))[0]
    
    min_fpr_to_idx = fpr[idx_of_maximum_tpr[0]]

    idx_to_use = idx_of_maximum_tpr[0]

    for idx in idx_of_maximum_tpr[1:]:
        if ((fpr[idx]< min_fpr_to_idx) and (fpr[idx]!=0)):
            min_fpr_to_idx = fpr[idx]
            idx_to_use = idx

    return ths[idx_to_use]


#################################################### GRAPHS #########################################################################################
#%%

def plot_single_exp_cm_analysis(path,output_path):

    cm_results = open_pickle_file(path)
    alphas = []
    ms=[]
    cc_val = []
    auc_val=[]
    conf_val= []

    
    for m in sorted(list(cm_results.keys())):
    
        for alpha in sorted(list(cm_results[m].keys())):
            alphas.append(alpha)
            ms.append(m)
            cc_value = cm_results[m][alpha]['cc']
            auc_value = cm_results[m][alpha]['auc']
            conf_value = cm_results[m][alpha]['cm_confidence']
        
            cc_val.append(cc_value)
            auc_val.append(auc_value)
            conf_val.append(conf_value)
    # saves dataframe  
    tmp_to_save = pd.DataFrame({'m':ms,'alpha':alphas,'pearson':cc_val,'AUC':auc_val,'confidence':conf_val})
    
    # saves images
    #save pearson
    plt.close()
    tmp_pearson = tmp_to_save.pivot(index="m",columns='alpha',values='pearson')
    sns.heatmap(tmp_pearson,cmap='coolwarm')
    plt.xlabel(chr(945))
    plt.title('$\\rho (\\mathcal{T}_0,\\mathcal{T}_{GT})$')
    plt.savefig(os.path.join(output_path,'pearson.pdf'),bbox_inches='tight')
    plt.close()
    #save AUC
    tmp_auc = tmp_to_save.pivot(index="m",columns='alpha',values='AUC')
    sns.heatmap(tmp_auc,cmap='coolwarm')
    plt.xlabel(chr(945))
    plt.title('CM ROC AUC')
    plt.savefig(os.path.join(output_path,'AUC.pdf'),bbox_inches='tight')
    plt.close()
    # save confidence 
    tmp_conf = tmp_to_save.pivot(index="m",columns='alpha',values='confidence')
    sns.heatmap(tmp_conf,cmap='coolwarm')
    plt.xlabel(chr(945))
    plt.title('$\\Gamma_{CM}$')
    plt.savefig(os.path.join(output_path,'confidence.pdf'),bbox_inches='tight')
    plt.close()

    

def plot_channel_prediction(analyzed_exp,stim_protocol,m,alpha,intensity,output_path):
    '''
    Plot all channel predictions.

    Input:
        - analyzed_exp: either the experiment class instance where data and params are stored or the path to the pickle file
        - stim_protocol: desired stim protocol for which channel prediction will be plotted
        - m: desired m parameter of MEArc model (among the used for training)
        - alpha: desired alpha parameter of MEArc model (among the used for training)
        - intensity: intenisty value of the stimulation
        - output_path: desired output path
    '''



    if isinstance(analyzed_exp,str):
        exp = open_pickle_file(analyzed_exp)

    else:
        exp = analyzed_exp
  
    int_time = exp.get_param('int_time')
    stim_elec_list = exp.get_data('stimulation')[stim_protocol][2]

    exp_type = exp.exp_type

    if exp_type=='sim':
        analyzed_stim_dict =exp.get_data('processed_stimulation')

    else:
        analyzed_stim_dict=exp.get_data('processed_filt_stimulation')


    norm_factor_back = exp.get_param('norm_factor_back')
    ground_truth_stim = analyzed_stim_dict[stim_protocol]['psth']/ norm_factor_back



    res = exp.get_data('predicted_psth')

    available_stim_protocols = list(res.keys())
    if stim_protocol not in available_stim_protocols:
        raise ValueError(f'desired stim protocol not found. Available are {available_stim_protocols}')
    available_m = list(res[available_stim_protocols[0]].keys())
    if m not in available_m:
        raise ValueError(f'desired m not found. Available are {available_m}')
    available_a = list(res[available_stim_protocols[0]][available_m[0]].keys())
    if alpha not in available_a:
        raise ValueError(f'desired alpha not found. Available are {available_a}')
    available_intensity = list(res[available_stim_protocols[0]][available_m[0]][available_a[0]].keys())
    if intensity not in available_intensity:
        raise ValueError(f'desired intensity not found. Available are {available_intensity}')

    res = res[stim_protocol][m][alpha][intensity]
    res_mean = res['mean']
    res_std = res['std']

    if exp_type=='sim':
        fig,axs = plt.subplots(12,5,sharex=True,sharey=True,figsize=(30,30),layout='constrained')
    
        axs = axs.flatten()
        for ch in range(res_mean.shape[0]):
                                
            ch_id = stim_elec_list[ch]
            
            axs[ch].errorbar(np.round(np.array(list(range(1,res_mean.shape[1]+1,1)))*int_time,1),res_mean[ch,:],yerr=res_std[ch,:],label = 'Prediction',color='red',marker='o',linestyle='--',linewidth=0.1)
            axs[ch].plot(np.array(list(range(1,res_mean.shape[1]+1,1)))*int_time,ground_truth_stim[ch,:],label='Ground truth',color='blue',marker='o',linestyle='--',linewidth=0.1)
            axs[ch].set_title(f'Channel {ch_id}')
            axs[ch].set_xticklabels([])
            axs[ch].set_yticklabels([])
        
        x_pos = plt.xlim()
        y_pos = plt.ylim()

        entry_legend_star = Line2D(xdata=[x_pos[1]-(x_pos[1]-x_pos[0])/8], ydata=[y_pos[1]-(y_pos[1]-y_pos[0])/8], color='yellow', marker='*',markersize=55, linestyle='None', label='Stimulated')
        
        axs_stim = stim_elec_list.index(str(stim_protocol))
        
        axs[axs_stim].add_line(entry_legend_star)
        

        handles, labels = axs[ch].get_legend_handles_labels()
        handles.extend([entry_legend_star]),labels.extend(['Stimulated'])
     
        lg = fig.legend(handles, labels,loc='center right',bbox_to_anchor=(1.3, 0.5))
        lg.legend_handles[0]._sizes = [60]
        lg.legend_handles[1]._sizes = [60]
        lg.legend_handles[2]._sizes = [60]
        #fig.suptitle('Channel response ISR',size=40)
        #fig.text(0.06, 0.5, 'ISR', ha='center', va='center', rotation='vertical',size=30)
        
        plt.savefig(os.path.join(output_path,f'channels_prediction.pdf'),bbox_inches='tight')
        plt.close()

    else:

        stim_elec_list = exp.get_data('filt_stimulation')[stim_protocol][2]
        unfiltered_elec_list = exp.get_data('stimulation')[stim_protocol][2]


        fig,axs = plt.subplots(8,8,sharex=True,sharey=True,figsize=(30,30),layout='constrained')
        
        tmp_x = None # those two are used just to save the coordinates of an axes having a plot so we can use them to retrieve labels to create the legend
        tmp_y = None 

        max_x = 0
        max_y = np.max(np.array(list(range(1,res_mean.shape[1]+1,1)))*int_time)

        for ch in range(len(unfiltered_elec_list)):


            ch_id = unfiltered_elec_list[ch]
            col_idx = int(ch_id[0])-1 # index of python
            row_idx = int(ch_id[1])-1

            
            if int(ch_id) in stim_elec_list:
                
                ch_to_plot_idx = stim_elec_list.index(int(ch_id))

                axs[row_idx,col_idx].errorbar(np.round(np.array(list(range(1,res_mean.shape[1]+1,1)))*int_time,1),res_mean[ch_to_plot_idx,:],yerr=res_std[ch_to_plot_idx,:],label = 'Prediction',color='red',marker='o',linestyle='--',linewidth=0.1)
                axs[row_idx,col_idx].plot(np.array(list(range(1,res_mean.shape[1]+1,1)))*int_time,ground_truth_stim[ch_to_plot_idx,:],label='Experimental',color='blue',marker='o',linestyle='--',linewidth=0.1)
                axs[row_idx,col_idx].set_title(f'Channel {ch_id}')
                axs[row_idx,col_idx].set_xticklabels([])
                axs[row_idx,col_idx].set_yticklabels([])

                check = np.max(res_mean[ch_to_plot_idx,:]+res_std[ch_to_plot_idx,:])
                if  check >max_y:
                    max_y= check 


                tmp_x = col_idx
                tmp_y = row_idx

        x_pos = plt.xlim()
        y_pos = plt.ylim()

        for ch in range(len(unfiltered_elec_list)):
                
            ch_id = unfiltered_elec_list[ch]
            col_idx = int(ch_id[0])-1 # index of python
            row_idx = int(ch_id[1])-1

            
            if int(ch_id) not in stim_elec_list:
            

                x_patch = Line2D(xdata=[x_pos[0],x_pos[1]], ydata=[y_pos[0],y_pos[1]],color='black',markersize=1000,markeredgewidth=10)
                axs[row_idx,col_idx].add_line(x_patch)
                x_patch = Line2D(xdata=[x_pos[0],x_pos[1]],ydata=[y_pos[1],y_pos[0]],color='black',markersize=1000,markeredgewidth=10)
                axs[row_idx,col_idx].add_line(x_patch)
                axs[row_idx,col_idx].set_title(f'Channel {ch_id}')
            
        
      
                    
        entry_legend_star = Line2D(xdata=[x_pos[1]-(x_pos[1]-x_pos[0])/8], ydata=[y_pos[1]-(y_pos[1]-y_pos[0])/8], color='yellow', marker='*',markersize=55, linestyle='None', label='Stimulated')
        
        col_idx_stim = int(str(stim_protocol)[0])-1 # index of python
        row_idx_stim = int(str(stim_protocol)[1])-1
        axs[row_idx_stim,col_idx_stim].add_line(entry_legend_star)
        

                                
        handles, labels = axs[tmp_y,tmp_x].get_legend_handles_labels()
        entry_legend_x = Line2D( [0], [0],color='black', marker='x',markersize=60,markeredgewidth=1,linestyle='None', label='Not analyzed')
        
        handles.extend([entry_legend_x,entry_legend_star]),labels.extend(['Not analyzed','Stimulated'])
     
        lg = fig.legend(handles, labels,loc='center right',bbox_to_anchor=(1.3, 0.5))
        lg.legend_handles[0]._sizes = [60]
        lg.legend_handles[1]._sizes = [60]
        lg.legend_handles[2]._sizes = [60]
        lg.legend_handles[3]._sizes = [60]
        #fig.suptitle('Channel response ISR',size=40)
        #fig.text(0.06, 0.5, 'ISR', ha='center', va='center', rotation='vertical',size=30)


     
        plt.savefig(os.path.join(output_path,f'channels_prediction.pdf'),bbox_inches='tight')
        plt.close()

        

    

    


   
def calculate_average_heatmap(response_xrmse_csv):

    '''
    Calculate the avarage of xrmse and auc heatmaps
    '''
    xrmse_list =[]
    auc_list = []
    pearson_list=[]
    
    m_values = None
    alpha_values = None

    for stim_name,stim_df in response_xrmse_csv.groupby(['stim']):

        xrmse_value_matrix = stim_df.pivot(index='alpha',columns='m',values='xrmse')

        
        m_values = xrmse_value_matrix.columns.values.tolist()
        alpha_values = xrmse_value_matrix.index.values.tolist()

        xrmse_value_matrix = xrmse_value_matrix.to_numpy()

        #max_value_xrmse = np.max(xrmse_value_matrix)

        #xrmse_value_matrix = xrmse_value_matrix/max_value_xrmse

        xrmse_list.append(xrmse_value_matrix)

        auc_value_matrix = stim_df.pivot(index='alpha',columns='m',values='auc')
        auc_list.append(auc_value_matrix.to_numpy())

        pearson_value_matrix = stim_df.pivot(index='alpha',columns='m',values='cc')
        pearson_list.append(pearson_value_matrix.to_numpy())

    auc_list= np.average(np.stack(auc_list),axis=0)
    xrmse_list= np.average(np.stack(xrmse_list),axis=0)
    pearson_list = np.average(np.stack(pearson_list),axis=0)

    return xrmse_list,auc_list,pearson_list,m_values,alpha_values


def retrieve_xrmse_across_stim(selected_m,response_xrmse_csv):

    xrmse_list = []
    auc_list = []
    pearson_list = []
    selected_alpha_list = []
    stim_list = []
    tau_list = []

    for stim_name,stim_df in response_xrmse_csv.groupby(['stim']):

        best_m_stim_df = stim_df[stim_df['m']==int(selected_m)]

        auc_value,alpha_idx = best_m_stim_df['auc'].max(),best_m_stim_df['auc'].argmax()

        selected_alpha,xrmse_value,tau_value,pearson_value = best_m_stim_df['alpha'].iloc[alpha_idx],best_m_stim_df['xrmse'].iloc[alpha_idx],best_m_stim_df['tau'].iloc[alpha_idx],best_m_stim_df['cc'].iloc[alpha_idx]

        pearson_list.append(pearson_value)
        xrmse_list.append(xrmse_value)
        auc_list.append(auc_value)
        stim_list.append(stim_name[0])
        selected_alpha_list.append(selected_alpha)
        tau_list.append(tau_value)
    
    return xrmse_list,auc_list,pearson_list,selected_alpha_list,stim_list,tau_list

def add_data_exp(df_to_add_data,xrmse_list,auc_list,pearson_list,selected_alpha_list,stim_list,tau_list,best_m,exp_name,quality_factor):


    for num in range(len(xrmse_list)):

        df_to_add_data['selected_m'].append(best_m)
        df_to_add_data['exp_name'].append(exp_name)
        df_to_add_data['q'].append(quality_factor)
        df_to_add_data['stim'].append(int(stim_list[num]))
        df_to_add_data['selected_alpha'].append(selected_alpha_list[num])
        df_to_add_data['xrmse'].append(xrmse_list[num])
        df_to_add_data['auc'].append(auc_list[num])
        df_to_add_data['cc'].append(pearson_list[num])
        df_to_add_data['tau'].append(tau_list[num])


def retrieve_cm_stim_data_to_plot(rc_model_analyzed_input_folder,spicodyn_results_csv_path,output_folder,sims=True):

    analyzed_rc_folder = [fold_path for fold_path in os.scandir(rc_model_analyzed_input_folder) if fold_path.is_dir()]



    stimulation_sum_up = {'exp_name':[],'q':[],'stim':[],'selected_m':[],'selected_alpha':[],'xrmse':[],'auc':[],'cc':[],'tau':[]}

    cm_sum_up = {'exp_name':[],'q':[],'selected_m':[],'selected_alpha':[],'pearson':[],'auc':[],'confidence':[]}

    cm_scatter_data = {'q':[],'metric_value':[],'metric':[]}

    cm_boxplot_data = {'algorithm':[],'pop':[],'metric':[],'metric_value':[]}
   
    avarage_map_folder = os.path.join(output_folder,'average_maps')
    if not os.path.exists(avarage_map_folder):
        os.makedirs(avarage_map_folder)


    for fold in analyzed_rc_folder:
        
        fold_specific_average_map_folder = os.path.join(avarage_map_folder,fold.name)
        if not os.path.exists(fold_specific_average_map_folder):
            os.makedirs(fold_specific_average_map_folder)
        
        # retrieve m and alpha from the response

        exp_info_file = os.path.join(fold,'analyzed_experiment.pickle') # pickle file of the exp
        
        # quality factor calculation
        exp_info_file = open_pickle_file(exp_info_file)
        tr_data = exp_info_file.get_data('training')
        pop_number = tr_data[0].shape[0] # retrieve population number
        training_point = 0
        for tr_window in tr_data:
            training_point += tr_window.shape[1]

        q_value = training_point/pop_number # quality value: it represents the goodness of the data, as the number of NBs is representative of the undergoing dynamic
    


        path_to_stim_data_fold = os.path.join(fold,'stimulus_prediction')

        path_to_xrmse_csv = os.path.join(path_to_stim_data_fold,'stimulus_prediction_best_intensity_data_auc_xrmse.csv')

        xrmse_csv = pd.read_csv(path_to_xrmse_csv,sep='\t')

    
        
        xrmse_avg,auc_avg,pearson_avg,m_values,alpha_values = calculate_average_heatmap(xrmse_csv)

        #save average maps
        plt.close()
        sns.heatmap(xrmse_avg,xticklabels=m_values, yticklabels=alpha_values)
        plt.xlabel('m')
        plt.ylabel(f'{chr(945)}')
        plt.savefig(os.path.join(fold_specific_average_map_folder,'average_xrmse.pdf'))
        plt.close()
        sns.heatmap(auc_avg,xticklabels=m_values, yticklabels=alpha_values)
        plt.xlabel('m')
        plt.ylabel(f'{chr(945)}')
        plt.savefig(os.path.join(fold_specific_average_map_folder,'average_auc.pdf'))
        plt.close()
        sns.heatmap(pearson_avg,xticklabels=m_values, yticklabels=alpha_values)
        plt.xlabel('m')
        plt.ylabel(f'{chr(945)}')
        plt.savefig(os.path.join(fold_specific_average_map_folder,'average_pearson.pdf'))
      

        #if np.any(np.isnan(pearson_avg)): 
        #    pearson_avg = np.where(np.isnan(pearson_avg),-1,pearson_avg)

        best_m_index = np.argwhere(auc_avg==np.max(auc_avg)) # retrieve the best xrmse value of the averaged heatmaps

        if len(best_m_index)>1:
            best_m_index = np.argwhere(xrmse_avg==np.min(xrmse_avg))[0][1]
        else:
            best_m_index=best_m_index[0][1]

        best_m = m_values[best_m_index]

        xrmse_list,auc_list,pearson_list,selected_alpha_list,stim_list,tau_list = retrieve_xrmse_across_stim(best_m,xrmse_csv)

        add_data_exp(stimulation_sum_up,xrmse_list,auc_list,pearson_list,selected_alpha_list,stim_list,tau_list,best_m,fold.name,q_value) # add the data to the stimulation_sum_up dictionary


        if sims == True:
            
           
            # retrieving file
            print(fold)
            path_to_cm_data = os.path.join(fold,'cm_analysis')
            
            cm_csv_file = os.path.join(path_to_cm_data,'cm_analysis.csv') # csv file with validation results used to choose best m and alpha

            
            tmp_df = pd.read_csv(cm_csv_file,sep='\t')

            best_m_df = tmp_df[tmp_df['m']==best_m]
            
            highest_idx =  np.argmax(best_m_df['pearson'])
        
            pearson_value = round(best_m_df['pearson'].iloc[highest_idx],5)
            auc_value = round(best_m_df['AUC'].iloc[highest_idx],5)
            alpha_value = best_m_df['alpha'].iloc[highest_idx]
            conf_value = round(best_m_df['confidence'].iloc[highest_idx],5)

            

            cm_scatter_data['metric_value'].append(auc_value)
            cm_scatter_data['metric_value'].append(pearson_value)
            cm_scatter_data['q'].append(q_value)
            cm_scatter_data['q'].append(q_value)
            cm_scatter_data['metric'].append('AUC')
            cm_scatter_data['metric'].append(f'{chr(961)}')


            # boxplot part
            pop_number = re.findall('\d*(?=_)',fold.name)[0]

            cm_boxplot_data['pop'].append(int(pop_number))
            cm_boxplot_data['pop'].append(int(pop_number))
    
            cm_boxplot_data['metric_value'].append(auc_value)
            cm_boxplot_data['metric'].append('AUC')
            cm_boxplot_data['metric_value'].append(pearson_value)
            cm_boxplot_data['metric'].append('Pearson')
            cm_boxplot_data['algorithm'].append('RC')
            cm_boxplot_data['algorithm'].append('RC')


            

            cm_sum_up['exp_name'].append(fold.name)
            cm_sum_up['q'].append(q_value)
            cm_sum_up['selected_m'].append(best_m)
            cm_sum_up['selected_alpha'].append(alpha_value)
            cm_sum_up['pearson'].append(pearson_value)
            cm_sum_up['auc'].append(auc_value)
            cm_sum_up['confidence'].append(conf_value) 


    if sims == True:

        spico_csv = pd.read_csv(spicodyn_results_csv_path,sep='\t')
        
        first_grouping = spico_csv.groupby(['exp_name'])

        for exp_name,exp_df in first_grouping:

            pop_number = re.findall('\d*(?=_)',exp_name[0])[0]
            

            second_grouping = exp_df.groupby(['algorithm'])

            for alg_name,alg_df in second_grouping:

                highest_idx = np.argmax(alg_df['Pearson'])
                pearson_value = round(alg_df['Pearson'].iloc[highest_idx],5)
                auc_value = round(alg_df['AUC'].iloc[highest_idx],5)

                

                cm_boxplot_data['metric_value'].append(pearson_value)
                cm_boxplot_data['metric'].append('Pearson')
                cm_boxplot_data['metric_value'].append(auc_value)
                cm_boxplot_data['metric'].append('AUC')
                cm_boxplot_data['algorithm'].append(alg_name[0])
                cm_boxplot_data['algorithm'].append(alg_name[0])
                cm_boxplot_data['pop'].append(int(pop_number))
                cm_boxplot_data['pop'].append(int(pop_number))


    stimulation_sum_up = pd.DataFrame(stimulation_sum_up)
    stimulation_sum_up.to_csv(os.path.join(output_folder,'stimulation_sum_up.csv'),sep='\t',index=False)

    if sims == True:

        cm_sum_up = pd.DataFrame(cm_sum_up)
        cm_sum_up.to_csv(os.path.join(output_folder,'cm_sum_up.csv'),sep='\t',index=False)

        
        cm_scatter_data = pd.DataFrame(cm_scatter_data)
        cm_scatter_data.head()
        cm_scatter_data.to_csv(os.path.join(output_folder,'cm_data_for_scatter.csv'),sep='\t',index=False)

        cm_boxplot_data = pd.DataFrame(cm_boxplot_data)
        cm_boxplot_data.to_csv(os.path.join(output_folder,'cm_data_for_boxplot.csv'),sep='\t',index=False)
   
    


def plot_cm_analysis(input_csv_scatter,output_folder):
    '''
    Function creates the plot of CM analysis
    '''

    tmp_df  = pd.read_csv(input_csv_scatter,sep='\t')
   
    auc_data = tmp_df[tmp_df['metric']=='AUC']
    pearson_data = tmp_df[tmp_df['metric']==f'{chr(961)}']

    res_model_auc = sm.OLS(auc_data['metric_value'], sm.add_constant(np.log(auc_data['q']))).fit()
    b_auc, m_auc= res_model_auc.params

    res_model_pearson = sm.OLS(pearson_data['metric_value'], sm.add_constant(np.log(pearson_data['q']))).fit()
    b_p, m_p= res_model_pearson.params
   
    plt.close()
    sns.set_theme(style="ticks")
    
    adjust_font_dimension(title_size=20,legend_size=20,label_size=35,x_ticks=25,y_ticks=25)
    tmp_graph = sns.lmplot(data=tmp_df,x='q',y='metric_value',fit_reg=True,logx=True,hue='metric',palette=['b','r'],markers=['*','.'],legend=False)
    
    plt.xscale('log')
    
    
    plt.plot(auc_data['q'].to_numpy(),(b_auc+ m_auc*(np.log(auc_data['q'].to_numpy()))),'b')
    plt.plot(pearson_data['q'].to_numpy(),(b_p+ m_p*(np.log(pearson_data['q'].to_numpy()))),'r')
    ptc = [Line2D( [0], [0],color='tab:blue',label='AUC',marker='*',markersize=15,linestyle=''),Line2D( [0], [0],color='tab:red',markersize=15,label=f'{chr(961)}',marker='o',linestyle='')]
    plt.legend(handles=ptc,bbox_to_anchor=(1.05, 0.5), loc='center left')
   
    plt.ylabel('Metrics')
    plt.xlabel('q')
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='center left',title=None)

    plt.savefig(os.path.join(output_folder,'auc_rho_vs_quality.pdf'),bbox_inches="tight")

   
    res_sum_auc = res_model_auc.summary(title='auc_vs_q_conn')
    res_sum_p = res_model_pearson.summary(title='pearson_vs_q_conn')



    with open(os.path.join(output_folder,'summary_auc_vs_q_connectivity.txt'), 'w+') as fh:
        fh.write(res_sum_auc.as_text())

    with open(os.path.join(output_folder,'summary_pearson_vs_q_connectivity.txt'), 'w+') as fh:
        fh.write(res_sum_p.as_text())

   



def box_plot_sim_analysis(input_csv_for_box,
                          output_path):
    
    
    final_df = pd.read_csv(input_csv_for_box,sep='\t')

    final_df['algorithm'].replace('NormCorr','CC',inplace=True)

    auc_rc = final_df[(final_df['algorithm']=='RC') & (final_df['metric']=='AUC')]['metric_value']
    print(auc_rc)
    auc_cc = np.array(final_df[(final_df['algorithm']=='CC') & (final_df['metric']=='AUC')]['metric_value'])
    auc_TE = np.array(final_df[(final_df['algorithm']=='TE') & (final_df['metric']=='AUC')]['metric_value'])

    p_rc = np.array(final_df[(final_df['algorithm']=='RC') & (final_df['metric']=='Pearson')]['metric_value'])
    p_cc = np.array(final_df[(final_df['algorithm']=='CC') & (final_df['metric']=='Pearson')]['metric_value'])
    p_TE = np.array(final_df[(final_df['algorithm']=='TE') & (final_df['metric']=='Pearson')]['metric_value'])



    kruskal_test_auc = kruskal(auc_cc,auc_rc,auc_TE)

    kruskal_test_p = kruskal(p_cc,p_rc,p_TE)

    print('auc',kruskal_test_auc)
    print('pearson',kruskal_test_p)






    pair_final_df = [[('RC','AUC'),('CC','AUC')],
                     [('RC','AUC'),('TE','AUC')],
                     
                     
                     [('RC','Pearson'),('CC','Pearson')],
                     [('RC','Pearson'),('TE','Pearson')],
                     ]
    


    plt.close()
    #sns.set_theme(style="ticks")
    plotting_parameters = {
    'data':    final_df,
    'x':       'algorithm',
    'y':       'metric_value',
    'hue':     'metric',
    'palette': ['b','r']
    }

    with sns.plotting_context():
        ax = plt.axes()

        
        #sns.boxplot(x='algorithm',y='metric_value',hue='metric',data=final_df,palette=['b','r'])
        sns.boxplot(**plotting_parameters)
        annotator = Annotator(ax=ax,pairs=pair_final_df,**plotting_parameters)
        annotator.configure(test='Mann-Whitney', verbose=True,comparisons_correction='Bonferroni').apply_and_annotate()
        sns.despine(offset=10, trim=True)
        plt.xlabel('')
        plt.ylabel('Metrics')
        plt.gca().set_ylim(top=1)
        plt.gca().spines.left.set_bounds(-0.5, 1)
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='center left',title=None)

        plt.savefig(os.path.join(output_path,'avg_auc_roc_alg_comparison.pdf'),bbox_inches="tight")
        plt.close()
    

    auc_df = final_df[final_df['metric']=='AUC']
    #sns.boxplot(x='algorithm',y='metric_value',hue='pop',data=auc_df,palette=['b','r','g','m','y'])
    #sns.despine(offset=10, trim=True)
    #plt.title('Algorithm comparison')
    #plt.ylabel('AUC')
    #plt.show()

    sns.boxplot(x='pop',y='metric_value',hue='algorithm',data=auc_df,palette=['b','r','g'])
    sns.despine(offset=10)
    plt.xlabel('Number of populations')
    plt.ylabel('AUC')
    plt.ylim(0,1)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='center left',title=None)
    plt.savefig(os.path.join(output_path,'auc_vs_population.pdf'),bbox_inches="tight")
    
    plt.close()
    #pearson_df = final_df[final_df['metric']=='Pearson']
    #sns.boxplot(x='algorithm',y='metric_value',hue='pop',data=pearson_df,palette=['b','r','g','m','y'])
    #sns.despine(offset=10, trim=True)
    #plt.title('Algorithm comparison')
    #plt.ylabel('Pearson')
    #plt.show()

    pearson_df = final_df[final_df['metric']=='Pearson']
    sns.boxplot(x='pop',y='metric_value',hue='algorithm',data=pearson_df,palette=['b','r','g'])
    sns.despine(offset=10, trim=True)
    
    plt.xlabel('Number of populations')
    plt.ylabel('$\\rho$')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='center left',title=None)
    plt.savefig(os.path.join(output_path,'pearson_vs_population.pdf'),bbox_inches="tight")
  
    plt.close()



def scatterplot_stim(stim_sum_up_csv_path,output_path,avg=True,log=False):

    stim_data = pd.read_csv(stim_sum_up_csv_path,sep='\t')

    out_name = '_vs_quality_stim.pdf'

    if avg == True:
        stim_data = stim_data.groupby(['exp_name']).aggregate({'auc':'mean','xrmse':'mean','q':'mean','cc':'mean'})
        out_name = '_vs_quality_stim_avg'

    # AUC
    avg_auc = stim_data['auc'].mean()
    
    l_08_l_100 = len(stim_data[(stim_data['auc']<avg_auc)&(stim_data['q']<100)]['auc'])
    l_08_m_100 = len(stim_data[(stim_data['auc']<avg_auc)&(stim_data['q']>=100)]['auc'])
    m_08_l_100 = len(stim_data[(stim_data['auc']>=avg_auc)&(stim_data['q']<100)]['auc'])
    m_08_m_100 = len(stim_data[(stim_data['auc']>=avg_auc)&(stim_data['q']>=100)]['auc'])
   
    df = pd.DataFrame({'q':['>= 100','< 100'],f'auc >= {avg_auc} ':[m_08_m_100,m_08_l_100],f'auc < {avg_auc}':[l_08_m_100,l_08_l_100]})
    df.to_csv(os.path.join(output_path,'auc_stim_graph_partition.csv'),sep='\t')

    plt.close()
    sns.set_theme(style="ticks")
    adjust_font_dimension(title_size=20,legend_size=30,label_size=35,x_ticks=25,y_ticks=25)
    fig, ax = plt.subplots()
    if log==True:
        ax.set_xscale('log')
        out_name=out_name+'_log'
    sns.scatterplot(data=stim_data,x='q',y='auc',**{"s": 70, "facecolor": "none", "linewidth": 1.5},edgecolor='black')
    ax.set_ylim(0.0,1.05)
    ax.axhline(y=avg_auc, color='grey', linestyle='--')
    ax.axvline(x = 100,color = 'grey',linestyle='--')
    #ax.axhline(y=0.5,color='red', linestyle='--')
    ax.axhspan(0,0.5,color='red',alpha=0.1)
    ax.set_ylim(0)

    plt.ylabel('AUC')
    plt.xlabel('q')
    plt.savefig(os.path.join(output_path,'auc'+out_name+'.pdf'),bbox_inches='tight')

    # XRMSE
    avg_xrmse = stim_data['xrmse'].mean()
    l_08_l_100 = len(stim_data[(stim_data['xrmse']<avg_xrmse)&(stim_data['q']<100)]['xrmse'])
    l_08_m_100 = len(stim_data[(stim_data['xrmse']<avg_xrmse)&(stim_data['q']>=100)]['xrmse'])
    m_08_l_100 = len(stim_data[(stim_data['xrmse']>=avg_xrmse)&(stim_data['q']<100)]['xrmse'])
    m_08_m_100 = len(stim_data[(stim_data['xrmse']>=avg_xrmse)&(stim_data['q']>=100)]['xrmse'])
   
    df = pd.DataFrame({'q':['>= 100','< 100'],f'xrmse >= {avg_xrmse} ':[m_08_m_100,m_08_l_100],f'xrmse < {avg_xrmse}':[l_08_m_100,l_08_l_100]})
    df.to_csv(os.path.join(output_path,'xrmse_stim_graph_partition.csv'),sep='\t')

    if avg == True:
        out_name = '_vs_quality_stim_avg.pdf'
    else:
        out_name = '_vs_quality_stim.pdf'

    plt.close()
    #sns.set_theme(style="ticks")
    fig, ax = plt.subplots()
    if log==True:
        ax.set_xscale('log')
        out_name='log'+out_name
    sns.scatterplot(data=stim_data,x='q',y='xrmse',**{"s": 70, "facecolor": "none", "linewidth": 1.5},edgecolor='black')
    ax.axhline(y=avg_xrmse, color='grey', linestyle='--')
    ax.axvline(x = 100,color = 'grey',linestyle='--')
   
    plt.ylabel('$\\overline{R}$')
    plt.xlabel('q')
    plt.savefig(os.path.join(output_path,'xrmse'+out_name),bbox_inches='tight')
    

    # pearson
    if avg == True:
        out_name = '_vs_quality_stim_avg.pdf'
    else:
        out_name = '_vs_quality_stim.pdf'
    plt.close()
    #sns.set_theme(style="ticks")
    fig, ax = plt.subplots()
    if log==True:
        ax.set_xscale('log')
        out_name='log'+out_name
    sns.scatterplot(data=stim_data,x='q',y='cc',**{"s": 70, "facecolor": "none", "linewidth": 1.5},edgecolor='black')
    plt.ylabel('$\\rho$')
    plt.xlabel('q')
    plt.savefig(os.path.join(output_path,'cc'+out_name),bbox_inches='tight')
    

    # if sim==True:
    #     plt.close()
    #     sns.set_theme(style="ticks")
    #     fig, ax = plt.subplots()
    #     ax.set_xscale('log')
    #     sns.regplot(data=stim_data,x='q',y='xrmse',fit_reg=True,logx=True,line_kws=dict(color="r"),ax=ax)


    #     plt.ylabel('$\\overline{R}$')
    #     plt.xlabel('q')

    #     res_model = sm.OLS(stim_data['xrmse'], sm.add_constant(np.log(stim_data['q']))).fit()
    #     b, m= res_model.params
    #     ax.plot(np.sort(stim_data['q'].to_numpy()),b+ m*(np.sort(np.log(stim_data['q'].to_numpy()))),'r')

    #     plt.savefig(os.path.join(output_path,'xrmse_vs_quality_stim.pdf'),bbox_inches='tight')
    #     res_sum = res_model.summary(title='xrmse_vs_q_regression')

    #     with open(os.path.join(output_path,'summary_xrmse_vs_q_stim.txt'), 'w+') as fh:
    #         fh.write(res_sum.as_text())



    #     plt.close()
    #     sns.set_theme(style="ticks")
    #     fig, ax = plt.subplots()
    #     ax.set_xscale('log')
    #     sns.regplot(data=stim_data,x='q',y='auc',fit_reg=True,logx=True,line_kws=dict(color="r"),ax=ax)


    #     plt.ylabel('AUC')
    #     plt.xlabel('q')

    #     res_model = sm.OLS(stim_data['auc'], sm.add_constant(np.log(stim_data['q']))).fit()
    #     b, m= res_model.params
    #     ax.plot(np.sort(stim_data['q'].to_numpy()),np.sort(b+ m*(np.log(stim_data['q'].to_numpy()))),'r')

    #     plt.savefig(os.path.join(output_path,'auc_vs_quality_stim.pdf'),bbox_inches='tight')
    #     res_sum = res_model.summary(title='auc_vs_q_regression')

    #     with open(os.path.join(output_path,'regression_auc_vs_q_stim.txt'), 'w+') as fh:
    #         fh.write(res_sum.as_text())
    # else:
    #     plt.close()
    #     sns.set_theme(style="ticks")
    #     fig, ax = plt.subplots()
    #     #ax.set_xscale('log')
    #     sns.regplot(data=stim_data,x='q',y='xrmse',fit_reg=True,logx=True,line_kws=dict(color="r"),ax=ax)


    #     plt.ylabel('$\\overline{R}$')
    #     plt.xlabel('q')

    #     res_model = sm.OLS(stim_data['xrmse'], sm.add_constant(np.log(stim_data['q']))).fit()
    #     b, m= res_model.params
    #     ax.plot(np.sort(stim_data['q'].to_numpy()),b+ m*(np.sort(np.log(stim_data['q'].to_numpy()))),'r')

    #     plt.savefig(os.path.join(output_path,'xrmse_vs_quality_stim.pdf'),bbox_inches='tight')
    #     res_sum = res_model.summary(title='xrmse_vs_q_regression')

    #     with open(os.path.join(output_path,'summary_xrmse_vs_q.txt'), 'w+') as fh:
    #         fh.write(res_sum.as_text())



    #     plt.close()
    #     sns.set_theme(style="ticks")
    #     fig, ax = plt.subplots()
    #     #ax.set_xscale('log')
    #     sns.regplot(data=stim_data,x='q',y='auc',fit_reg=True,logx=True,line_kws=dict(color="r"),ax=ax)


    #     plt.ylabel('AUC')
    #     plt.xlabel('q')

    #     res_model = sm.OLS(stim_data['auc'], sm.add_constant(np.log(stim_data['q']))).fit()
    #     b, m= res_model.params
    #     ax.plot(np.sort(stim_data['q'].to_numpy()),b+ m*(np.sort(np.log(stim_data['q'].to_numpy()))),'r')

    #     plt.savefig(os.path.join(output_path,'auc_vs_quality_stim.pdf'),bbox_inches='tight')
    #     res_sum = res_model.summary(title='auc_vs_q_regression')

    #     with open(os.path.join(output_path,'regression_auc_vs_q.txt'), 'w+') as fh:
    #         fh.write(res_sum.as_text())




def boxplot_stim(stim_sum_up_csv_path,output_path):

    stim_data = pd.read_csv(stim_sum_up_csv_path,sep='\t')

    
    to_change = stim_data['exp_name'].tolist()

    pop_number = [int(re.findall('\d*(?=_)',exp_name)[0]) for exp_name in to_change]

    stim_data_copy = copy.deepcopy(stim_data)

    stim_data_copy['exp_name']=pop_number
    


    plt.close()
    sns.boxplot(data=stim_data_copy,x='exp_name',y='xrmse',palette=['r'])
    sns.despine(offset=10, trim=True)
    plt.xlabel('Number of populations')
    plt.ylabel('$\\overline{R}$')
    plt.savefig(os.path.join(output_path,'xrmse_vs_population_stim.pdf'),bbox_inches="tight")
    



    plt.close()
    sns.boxplot(data=stim_data_copy,x='exp_name',y='auc',palette=['b'])
    sns.despine(offset=10, trim=True)
    plt.xlabel('Number of populations')
    plt.ylabel('AUC')
    plt.savefig(os.path.join(output_path,'auc_vs_population_stim.pdf'),bbox_inches="tight")
    
    plt.close()





def mix_cm_stim_graph(stim_sum_up_csv_path,cm_sum_up_path,output_path):

    cm_df = pd.read_csv(cm_sum_up_path,sep='\t')
    cm_df.rename(columns={'auc':'auc_conn'},inplace=True)

    stim_df = pd.read_csv(stim_sum_up_csv_path,sep='\t')
    stim_df = stim_df.groupby('exp_name').agg({'xrmse':'mean','auc':'mean'})

    merged_df = cm_df.merge(stim_df,left_on='exp_name',right_on='exp_name')

    # pearson vs xrmse

    plt.close()
    fig, ax = plt.subplots()
    sns.regplot(data=merged_df,x='pearson',y='xrmse',fit_reg=True,line_kws=dict(color="r"))
    plt.ylabel('$\\overline{R}$')
    plt.xlabel('$\\rho$')
    
    res_model = sm.OLS(merged_df['xrmse'], sm.add_constant(merged_df['pearson'])).fit()
    b, m= res_model.params
    ax.plot(merged_df['pearson'].to_numpy(),(b+ (m*merged_df['pearson'].to_numpy())),'r')

    plt.savefig(os.path.join(output_path,'pearson_vs_xrmse.pdf'),bbox_inches='tight')
    res_sum = res_model.summary(title='pearson_vs_xrmse_regression')

    with open(os.path.join(output_path,'pearson_vs_xrmse.txt'), 'w+') as fh:
        fh.write(res_sum.as_text())
    plt.close()

    # auc_conn vs auc

    fig, ax = plt.subplots()
    sns.regplot(data=merged_df,x='auc_conn',y='auc',fit_reg=True,line_kws=dict(color="r"))
    plt.xlabel('CM AUC')
    plt.ylabel('Response AUC')
    plt.ylim(0,1)
    
    
    res_model = sm.OLS(merged_df['auc'], sm.add_constant(merged_df['auc_conn'])).fit()
    b, m= res_model.params
    ax.plot(merged_df['auc_conn'].to_numpy(),(b+ (m*merged_df['auc_conn'].to_numpy())),'r')

    plt.savefig(os.path.join(output_path,'auc_conn_vs_auc_resp.pdf'),bbox_inches='tight')
    res_sum = res_model.summary(title='auc_conn_vs_auc_resp')

    with open(os.path.join(output_path,'auc_conn_vs_auc_resp.txt'), 'w+') as fh:
        fh.write(res_sum.as_text())


    # auc_conn vs auc
    plt.close()
    fig, ax = plt.subplots()
    sns.regplot(data=merged_df,x='pearson',y='auc',fit_reg=True,line_kws=dict(color="r"))
    plt.xlabel('$\\rho$')
    plt.ylabel('Response AUC')
    
    res_model = sm.OLS(merged_df['auc'], sm.add_constant(merged_df['pearson'])).fit()
    b, m= res_model.params
    ax.plot(merged_df['pearson'].to_numpy(),(b+ (m*merged_df['pearson'].to_numpy())),'r')

    plt.savefig(os.path.join(output_path,'pearson_vs_auc_resp.pdf'),bbox_inches='tight')
    res_sum = res_model.summary(title='pearson_vs_auc_resp')

    with open(os.path.join(output_path,'pearson_vs_auc_resp.txt'), 'w+') as fh:
        fh.write(res_sum.as_text())



def final_results_table(stim_sum_up_csv_path,conn_sum_up_csv_path,output_path,sim=False):

    final_stim_res = {'experiments_number':[],'avg_exp_auc':[],'std_exp_auc':[],'avg_exp_xrmse':[],'std_exp_xrmse':[],'avg_exp_cc':[],'std_exp_cc':[],
                      'protocols_number':[],'avg_prot_auc':[],'std_prot_auc':[],'avg_prot_xrmse':[],'std_prot_xrmse':[],'avg_prot_cc':[],'std_prot_cc':[]}

    # data considering each stimulation protocol as indipendent
    stim_data = pd.read_csv(stim_sum_up_csv_path,sep='\t')

    protocols_number = len(stim_data)

    avg_auc_protocol = stim_data['auc'].mean()
    std_auc_protocol = stim_data['auc'].std()

    avg_xrmse_protocol = stim_data['xrmse'].mean()
    std_xrmse_protocol = stim_data['xrmse'].std()

    avg_cc_protocol = stim_data['cc'].mean()
    std_cc_protocol = stim_data['cc'].std()

    final_stim_res['protocols_number'].append(protocols_number),final_stim_res['avg_prot_auc'].append(avg_auc_protocol),final_stim_res['std_prot_auc'].append(std_auc_protocol)
    final_stim_res['avg_prot_xrmse'].append(avg_xrmse_protocol),final_stim_res['std_prot_xrmse'].append(std_xrmse_protocol)
    final_stim_res['avg_prot_cc'].append(avg_cc_protocol),final_stim_res['std_prot_cc'].append(std_cc_protocol)


    # average of all stimulation protocols for each experiment 
    stim_data = stim_data.groupby(['exp_name']).aggregate({'auc':'mean','xrmse':'mean','q':'mean','cc':'mean'})
    
    experiments_number = len(stim_data)

    avg_auc_exp = stim_data['auc'].mean()
    std_auc_exp = stim_data['auc'].std()

    avg_xrmse_exp = stim_data['xrmse'].mean()
    std_xrmse_exp = stim_data['xrmse'].std()

    avg_cc_exp = stim_data['cc'].mean()
    std_cc_exp = stim_data['cc'].std()

    
    final_stim_res['experiments_number'].append(experiments_number),final_stim_res['avg_exp_auc'].append(avg_auc_exp),final_stim_res['std_exp_auc'].append(std_auc_exp)
    final_stim_res['avg_exp_xrmse'].append(avg_xrmse_exp),final_stim_res['std_exp_xrmse'].append(std_xrmse_exp)
    final_stim_res['avg_exp_cc'].append(avg_cc_exp),final_stim_res['std_exp_cc'].append(std_cc_exp)

    final_stim_res = pd.DataFrame(final_stim_res)

    final_stim_res.to_csv(os.path.join(output_path,'final_results_table_stimulation.csv'),sep='\t')


    if sim==True:

        conn_data = pd.read_csv(conn_sum_up_csv_path,sep='\t')

        df = {'simulation_number':[],'avg_auc':[],'std_auc':[],'avg_pearson':[],'std_pearson':[],'avg_confidence':[],'std_confidence':[]}

        sim_number = len(conn_data)

        avg_auc = conn_data['auc'].mean()
        std_auc = conn_data['auc'].std()

        avg_pear = conn_data['pearson'].mean()
        std_pear = conn_data['pearson'].std()

        avg_confidence = conn_data['confidence'].mean()
        std_confidence = conn_data['confidence'].std()


        df['simulation_number'].append(sim_number)
        df['avg_auc'].append(avg_auc),df['std_auc'].append(std_auc)
        df['avg_pearson'].append(avg_pear),df['std_pearson'].append(std_pear)
        df['avg_confidence'].append(avg_confidence),df['std_confidence'].append(std_confidence)

        df = pd.DataFrame(df)
        df.to_csv(os.path.join(output_path,'final_results_table_connectivity.csv'),sep='\t')

