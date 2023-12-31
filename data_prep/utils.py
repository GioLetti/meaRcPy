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
from scipy.stats import ttest_ind,linregress
from matplotlib.ticker import MultipleLocator



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
    

#%%
def save_pickle_file(object, # object to save
                     path:os.path): # path where to save
    '''
    Function saves the object as pickle file
    '''

    with open(path, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def open_pickle_file(pickel_file_path:str
                    ):
    '''
    function open a pickle file
    '''

    pkl_file = open(pickel_file_path,'rb')
    open_pickle = pickle.load(pkl_file)

    return open_pickle

def take_best_m_alpha(validationf_file_path):

    df = pd.read_csv(validationf_file_path,sep='\t')
    best_idx = df['avg_rmse'].argmin()

    alpha = df['alpha'][best_idx] # optimal alpha
    m = df['m'][best_idx] # optimal m

    return m,alpha




def plot_mea_map(unfiltered_channel_list,filtered_channel_list,channel_integral,stim_electrode,title,output_path,bin=False,bin_err = False):

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

    entry_legend_x = Line2D( [0], [0],color='black', marker='x',markersize=15,markeredgewidth=1,linestyle='None', label='Not analyzed')
    entry_legend_star = Line2D([0], [0], color='yellow', marker='*',markersize=15, linestyle='None', label='Stimulated')
    

    if bin==True:
        cmap, norm = from_levels_and_colors([0, 0.5, 1.1], ['tab:blue', 'r'])

        
        # create a patch (proxy artist) for every color 
        patches = [mpatches.Patch(color='tab:blue',label = 'Inactive'),mpatches.Patch(color='r',label = 'Active'),entry_legend_x,entry_legend_star]
        

        plt.close()

        fig,ax = plt.subplots()
        ax.scatter(x=y_pos_to_annotate,y=x_pos_to_annotate,marker='x',s=1000,c='black')
        ax.scatter(x = [y_stim],y=[x_stim],marker='*',s=500,c='yellow')
        ax.imshow(mea_map,interpolation='none', cmap=cmap, norm=norm)
        ax.set_xticklabels(np.arange(0,9,1))
        ax.set_yticklabels(np.arange(0,9,1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.grid(True,which='minor',color='black', linestyle='-', linewidth=1)
        ax.legend(handles=patches,bbox_to_anchor=(1.45, 1))
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
        ax.set_xticklabels(np.arange(0,9,1))
        ax.set_yticklabels(np.arange(0,9,1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.grid(True,which='minor',color='black', linestyle='-', linewidth=1)
        ax.legend(handles=patches,bbox_to_anchor=(1.45, 1))
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
        ax.set_xticklabels(np.arange(0,9,1))
        ax.set_yticklabels(np.arange(0,9,1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.grid(True,which='minor',color='black', linestyle='-', linewidth=1)
        ax.legend(handles=[entry_legend_x,entry_legend_star],bbox_to_anchor=(1.75, 1))
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
    
    sns.lmplot(data=tmp_df,x='q',y='metric_value',fit_reg=True,logx=True,hue='metric',palette=['b','r'],markers=['*','.'])
    plt.xscale('log')
    
    
    plt.plot(auc_data['q'].to_numpy(),(b_auc+ m_auc*(np.log(auc_data['q'].to_numpy()))),'b')
    plt.plot(pearson_data['q'].to_numpy(),(b_p+ m_p*(np.log(pearson_data['q'].to_numpy()))),'r')
    plt.ylabel('')
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

    

    pair_final_df = [[('RC','AUC'),('CC','AUC')],
                     [('RC','AUC'),('TE','AUC')],
                     
                     
                     [('RC','Pearson'),('CC','Pearson')],
                     [('RC','Pearson'),('TE','Pearson')],
                     ]
    


    plt.close()
    sns.set_theme(style="ticks")
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
        annotator.configure(test='t-test_ind', verbose=True).apply_and_annotate()
        sns.despine(offset=10, trim=True)
        plt.xlabel('')
        plt.ylabel('')
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
    sns.despine(offset=10, trim=True)
    plt.xlabel('Number of populations')
    plt.ylabel('AUC')
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



def scatterplot_stim(stim_sum_up_csv_path,output_path,sim=True):

    stim_data = pd.read_csv(stim_sum_up_csv_path,sep='\t')

    
    plt.close()
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    sns.scatterplot(data=stim_data,x='q',y='auc')


    plt.ylabel('AUC')
    plt.xlabel('q')

    
    plt.savefig(os.path.join(output_path,'auc_vs_quality_stim.pdf'),bbox_inches='tight')
    

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
    plt.xlabel('$\\rho')
    plt.ylabel('Response AUC')
    
    res_model = sm.OLS(merged_df['auc'], sm.add_constant(merged_df['pearson'])).fit()
    b, m= res_model.params
    ax.plot(merged_df['pearson'].to_numpy(),(b+ (m*merged_df['pearson'].to_numpy())),'r')

    plt.savefig(os.path.join(output_path,'pearson_vs_auc_resp.pdf'),bbox_inches='tight')
    res_sum = res_model.summary(title='pearson_vs_auc_resp')

    with open(os.path.join(output_path,'pearson_vs_auc_resp.txt'), 'w+') as fh:
        fh.write(res_sum.as_text())

