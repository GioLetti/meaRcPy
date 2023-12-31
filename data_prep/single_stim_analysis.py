#%%
import numpy as np
import pandas as pd
import pickle
import os
#import yaml
import matplotlib.pyplot as plt
from typing import Union
import seaborn as sns
from data_prep.utils import open_pickle_file,save_pickle_file
from data_prep.analyzer import analyzer
import math
from sklearn import metrics
import copy
#%%

def fix_roc(binarized_gt_tmp,amplitude_psth,single_pred_path,gt_thr):
     
    if (np.sum(binarized_gt_tmp)/len(binarized_gt_tmp)) == 1:
        
        binarized_gt_tmp = np.append(binarized_gt_tmp,0) # to avoid zero division in roc
        amplitude_psth = np.append(amplitude_psth,(np.max(amplitude_psth)+0.01)/2)
        roc_curve_path = os.path.join(single_pred_path,'Response_ROC(.pdf')

    elif (np.sum(binarized_gt_tmp)/len(binarized_gt_tmp)) == 0:

        binarized_gt_tmp = np.append(binarized_gt_tmp,1) # to avoid zero division in roc
        amplitude_psth = np.append(gt_thr+0.01)
        roc_curve_path = os.path.join(single_pred_path,'Response_ROC(.pdf')
    else:

        roc_curve_path = os.path.join(single_pred_path,'Response_ROC.pdf')

    return binarized_gt_tmp,amplitude_psth,roc_curve_path




def plot_mea_map(unfiltered_channel_list,filtered_channel_list,channel_integral,stim_electrode,title,output_path):

    mea_map = np.zeros((8,8))
    mea_map[:]=np.nan
    
    for ch in range(channel_integral.shape[0]):

        ch_idx = filtered_channel_list[ch]
        col_idx = int(ch_idx[0])-1 # index of python
        row_idx = int(ch_idx[1])-1

        mea_map[row_idx,col_idx] = channel_integral[ch]

    

    x_pos_to_annotate = [4]
    y_pos_to_annotate = [0]

    for ch in range(len(unfiltered_channel_list)):

        ch_idx = unfiltered_channel_list[ch]

        if ch_idx not in filtered_channel_list:

            col_idx = int(ch_idx[0])-1 # index of python
            row_idx = int(ch_idx[1])-1

            y_pos_to_annotate.append(col_idx)
            x_pos_to_annotate.append(row_idx)

    stim_electrode = str(stim_electrode)
    x_stim = int(stim_electrode[1])-1
    y_stim = int(stim_electrode[0])-1

    plt.close()
    
    plt.scatter(x=y_pos_to_annotate,y=x_pos_to_annotate,marker='x',s=1000,c='black')
    plt.scatter(x = [y_stim],y=[x_stim],marker='*',s=500,c='yellow')
    plt.imshow(mea_map,cmap='coolwarm')
    plt.colorbar()
    
    plt.xticks(np.arange(0,8,1),labels=np.arange(1,9,1))
    plt.yticks(np.arange(0,8,1),labels=np.arange(1,9,1))
    plt.grid(False)
    plt.title(title)
    plt.savefig(output_path,bbox_inches='tight')
    plt.close()



def full_characterization_stim(path_to_exp_folder:str,
                            gt_threshold:float=0.5):
    
    if isinstance(path_to_exp_folder,str):

        exp = open_pickle_file(os.path.join(path_to_exp_folder,'analyzed_experiment.pickle'))

        int_time = exp.get_param('int_time')

        tmp_process = exp.get_data('processed_stimulation') # in case the experiment is a simulation
        if tmp_process == None:
            processed_stim = exp.get_data('processed_filt_stimulation') # in case it is an experimental recording
        else:
            processed_stim = tmp_process

        all_mearc_model = open_pickle_file(os.path.join(path_to_exp_folder,'mea_optimal.pickle'))


    # create folder to store prediction data of all mearc models
    pred_folder_path = os.path.join(path_to_exp_folder,'all_mearc_stimulus_prediction_1')
    if not os.path.exists(pred_folder_path):
        os.makedirs(pred_folder_path)

    exp.results_path = pred_folder_path

    anl = analyzer(exp)

    #anl.stimulate_mearc(all_mearc_model,processed_stim,20,data_name='all_mearc_predicted_stim')

    #predicted_stim = exp.get_data('all_mearc_predicted_stim')

    #save_pickle_file(predicted_stim,os.path.join(pred_folder_path,'full_predicted_stimulation.pickle'))

    path_to_remove = os.path.join(path_to_exp_folder,'all_mearc_stimulus_prediction')

    predicted_stim = open_pickle_file(os.path.join(path_to_remove,'full_predicted_stimulation.pickle'))


    stims_final_xrmse_csv = {'stim':[],'m':[],'alpha':[],'intensity':[],'xrmse':[],'auc':[],'tau':[]}
    stims_final_auc_csv = {'stim':[],'m':[],'alpha':[],'intensity':[],'xrmse':[],'auc':[],'tau':[]}

    for stim in predicted_stim:

        #all_channel_pred_stim_m = os.path.join(pred_folder_path,f'prediction_stim_{stim}')
        single_stim_path = os.path.join(pred_folder_path,f'prediction_stim_{stim}')

        if not os.path.exists(single_stim_path):
            os.makedirs(single_stim_path)
        
        stim_channel_int_folder = os.path.join(single_stim_path,'channel_integral_maps')
        if not os.path.exists(stim_channel_int_folder):
            os.makedirs(stim_channel_int_folder)

        if exp.exp_type =='exp':
            best_channel_int_prediction =None # variable where to store the channel integral of the best prediction based on xrmse
            best_m_channel_int_prediction = None # same but for m
            best_alpha_channel_int_prediction = None # same but for alpha
            best_int_channel_int_prediction = None # same for intensity
            best_overall_xrmse_channel_int_prediction = math.inf # used to identify the best results
            ch_wise_error_int_prediction = None
            best_channel_int_bin_prediction = None # variable where to store the channel integral of the best prediction based on xrmse


        tmp_pred = predicted_stim[stim]

        gt_integral = processed_stim[stim]['integral']
        norm_factor_stim = processed_stim[stim]['norm_factor_stim']
        ground_truth_stim = processed_stim[stim]['psth']/ norm_factor_stim
        gt_thr = gt_threshold/norm_factor_stim

        binarized_gt = np.where(np.sum(np.where(ground_truth_stim>=gt_thr,1,0),axis=1)>=1,1,0)

        if (np.sum(binarized_gt)/len(binarized_gt)) == 0:
            ground_truth_stim = processed_stim[stim]['psth']/1

        
        if exp.exp_type == 'exp':
            stim_elec_list = exp.get_data('filt_stimulation')[stim][2]
            unfiltered_elec_list = exp.get_data('stimulation')[stim][2]
        else:
            stim_elec_list = exp.get_data('stimulation')[stim][2]
        
        all_m_ch_prediction_dict = {}

        all_m_dict = {}

        if exp.exp_type == 'exp':
        
            plot_mea_map(unfiltered_elec_list,stim_elec_list,gt_integral,stim,'Response integral [a.u.]',os.path.join(stim_channel_int_folder,f'stim_{stim}_gt_integral.pdf'))
            plot_mea_map(unfiltered_elec_list,stim_elec_list,binarized_gt,stim,'Response integral [a.u.]',os.path.join(stim_channel_int_folder,f'stim_{stim}_binarized_gt_integral.pdf'))
        

        m_values = sorted(list(tmp_pred.keys()))

        #heatmap m vs alpha auc and xrmse for best intensity 

        heatmap_xrmse_xrmse = {'m':[],'alpha':[],'best_intensity':[],'xrmse':[]}
        heatmap_auc_xrmse = {'m':[],'alpha':[],'best_intensity':[],'auc':[]}
        heatmap_xrmse_auc = {'m':[],'alpha':[],'best_intensity':[],'xrmse':[]}
        heatmap_auc_auc = {'m':[],'alpha':[],'best_intensity':[],'auc':[]}


        for m in m_values:
            
            m_path = os.path.join(single_stim_path,f'mearc_m_{m}_prediction') # prediction folder of specific m

            all_m_ch_prediction_dict[m] ={}
            all_m_dict[m]={}

            tmp_m_pred = tmp_pred[m]

            alphas_values = sorted(list(tmp_m_pred.keys())) # retrieve all alphas value

            # for merged alpha graphs
            all_alpha_dict = {}
            all_alpha_dict_prediction ={}

            #all_channel_pred_stim_path = os.path.join(m_path,f'prediction_stim_{stim}_m_{m}_all_alpha')
            all_alpha_path = os.path.join(m_path,'all_alpha') # folder where graphs showing all alphas are saved

            os.makedirs(all_alpha_path)


            for alpha_val in alphas_values:

                all_alpha_dict[alpha_val]={}

                all_alpha_dict_prediction[alpha_val] ={}

                alpha_value_specific_path = os.path.join(m_path,f'alpha_{alpha_val}') # path for a specific alpha value

                all_m_dict[m][alpha_val]={}
                all_m_ch_prediction_dict[m][alpha_val]={}

                if not os.path.exists(alpha_value_specific_path):
                    os.makedirs(alpha_value_specific_path)

                tmp_pred_alpha = tmp_m_pred[alpha_val]

                intensity_values = sorted(list(tmp_pred_alpha.keys()))

                overall_err_list = []
                overall_tau_list = []
                auc_list=[]

                for intensity in intensity_values:


                    predicted_psth = predicted_stim[stim][m][alpha_val][intensity]['mean']

                    pred_integral = anl._PSTH_integral(predicted_psth)

                    _,_,overall_error,overall_tau = anl._calculate_prediction_error(ground_truth_stim,predicted_psth,gt_integral,pred_integral)


                    max_amplitude_psth_roc = np.max(predicted_psth,axis=1)
                    binarized_gt_tmp = copy.deepcopy(binarized_gt)
                    binarized_gt_roc,predicted_psth_roc,_ = fix_roc(binarized_gt_tmp,max_amplitude_psth_roc,'_',gt_thr=gt_thr)


                    fpr, tpr, _ = metrics.roc_curve(binarized_gt_roc, predicted_psth_roc)
                    roc_auc = metrics.auc(fpr, tpr)

                    overall_err_list.append(overall_error)
                    overall_tau_list.append(overall_tau)
                    auc_list.append(roc_auc)

                all_alpha_dict[alpha_val]['xrmse']=overall_err_list
                all_alpha_dict[alpha_val]['auc']=auc_list


                all_m_dict[m][alpha_val]['xrmse']=overall_err_list
                all_m_dict[m][alpha_val]['auc']=auc_list



                #overall_error vs intensity
                plt.close()
                plt.style.use('seaborn-v0_8')
                plt.plot(intensity_values,overall_err_list)
                
                plt.xlabel('Intensity [a.u.]')
                plt.ylabel('$\\overline{R}$')
                
                plt.savefig(os.path.join(alpha_value_specific_path,'xrmse_vs_intensity.pdf'))
                plt.close()

                #overall_tau vs intensity
                plt.close()
                plt.style.use('seaborn-v0_8')
                plt.plot(intensity_values,overall_tau_list)
                
                plt.xlabel('Intensity [a.u.]')
                plt.ylabel('$\\overline{\\tau}_l [ms]$')
                
                plt.savefig(os.path.join(alpha_value_specific_path,f'{chr(964)}_vs_intensity.pdf'))
                plt.close()

                #auc vs intensity
                plt.close()
                plt.style.use('seaborn-v0_8')
                plt.plot(intensity_values,auc_list)
                
                plt.xlabel('Intensity [a.u.]')
                plt.ylabel(f'AUC')
                
                plt.savefig(os.path.join(alpha_value_specific_path,f'Response_auc_vs_intensity.pdf'))
                plt.close()


                # Analysis of best intensity using AUC
                idx_best_intensity = np.argmax(auc_list) # index of the intenisty with  highest auc
                best_intensity = intensity_values[idx_best_intensity] # best intensity

                best_intensity_psth = predicted_stim[stim][m][alpha_val][best_intensity]['mean']
                best_intensity_psth_std = predicted_stim[stim][m][alpha_val][best_intensity]['std']
                best_intensity_integral = anl._PSTH_integral(best_intensity_psth)
                
                ch_wise_error,_,overall_error,overall_tau = anl._calculate_prediction_error(ground_truth_stim,best_intensity_psth,gt_integral,best_intensity_integral,plot=False,path_to_save=None)                
                

                tmp_auc_path = os.path.join(pred_folder_path,'best_int_auc')
                if not os.path.exists(tmp_auc_path):
                    os.makedirs(tmp_auc_path)

                plt.close()
                sns.heatmap(best_intensity_psth,cmap='coolwarm',xticklabels=np.round(np.array(list(range(1,best_intensity_psth.shape[1]+1,1)))*int_time,1),yticklabels=np.array(list(range(1,best_intensity_psth.shape[0]+1,1))))
                
                plt.title(f'Normalized ISR - Prediction [a.u.], Time bin = {int_time} ms')
                plt.xlabel('Time bin [ms]')
                plt.ylabel('Channel number')
                plt.savefig(os.path.join(tmp_auc_path,f'psth_stim_protocol_{stim}_m_{m}_{chr(945)}_{alpha_val}_intensity_{best_intensity}_AUC.pdf'))
                plt.close()


                channel_prediction_folder = os.path.join(alpha_value_specific_path,'channel_prediction_auc')
                os.makedirs(channel_prediction_folder)
                
                # for ch in range(best_intensity_psth.shape[0]):
                    
                #     ch_id = stim_elec_list[ch]
                #     plt.close()
                #     plt.style.use('seaborn-v0_8')
                #     plt.errorbar(np.round(np.array(list(range(1,best_intensity_psth.shape[1]+1,1)))*int_time,1),best_intensity_psth[ch,:],yerr=best_intensity_psth_std[ch,:],label = 'Prediction',color='red',marker='o',linestyle='--',linewidth=0.5)
                #     plt.plot(np.array(list(range(1,best_intensity_psth.shape[1]+1,1)))*int_time,ground_truth_stim[ch,:],label='Ground truth',color='blue',marker='o',linestyle='--',linewidth=0.5)
                #     plt.title(f'channel {ch_id} response ISR, Time bin = {int_time} ms')
                #     plt.xlabel(f'Time (ms)')
                #     plt.ylabel('ISR [a.u.]')
                #     plt.legend()
                #     plt.savefig(os.path.join(channel_prediction_folder,f'channel_{ch_id}_prediction.pdf'))
                #     plt.close()

                # response roc
                max_amplitude_psth = np.max(best_intensity_psth,axis=1)

                binarized_gt_tmp_2 = copy.deepcopy(binarized_gt)

                binarized_gt_roc_2,max_amplitude_psth,roc_curve_path = fix_roc(binarized_gt_tmp_2,max_amplitude_psth,alpha_value_specific_path,gt_thr=gt_thr)
                
                auc = anl._roc_curve(binarized_gt_roc_2,max_amplitude_psth,f'm={m},{chr(945)}={alpha_val},intensity={best_intensity}',roc_curve_path)

                heatmap_xrmse_auc['m'].append(m),heatmap_xrmse_auc['alpha'].append(alpha_val),heatmap_xrmse_auc['best_intensity'].append(best_intensity),heatmap_xrmse_auc['xrmse'].append(overall_error)
                
                heatmap_auc_auc['m'].append(m),heatmap_auc_auc['alpha'].append(alpha_val),heatmap_auc_auc['best_intensity'].append(best_intensity),heatmap_auc_auc['auc'].append(auc)

                
                
                stims_final_auc_csv['stim'].append(stim),stims_final_auc_csv['m'].append(m),stims_final_auc_csv['alpha'].append(alpha_val)
                stims_final_auc_csv['intensity'].append(best_intensity),stims_final_auc_csv['xrmse'].append(overall_error),stims_final_auc_csv['auc'].append(auc),stims_final_auc_csv['tau'].append(overall_tau)

                
                # Analysis of best intensity using XRMSE
                idx_best_intensity = np.argmin(overall_err_list) # index of the intenisty with lowest overall error
                best_intensity = intensity_values[idx_best_intensity] # best intensity

                best_intensity_psth = predicted_stim[stim][m][alpha_val][best_intensity]['mean']
                best_intensity_psth_std = predicted_stim[stim][m][alpha_val][best_intensity]['std']
                best_intensity_integral = anl._PSTH_integral(best_intensity_psth)

                tau_xrmse_path = os.path.join(alpha_value_specific_path,'channels_tau_xrmse')
                if not os.path.exists(tau_xrmse_path):
                    os.makedirs(tau_xrmse_path)

                ch_wise_error,_,overall_error,overall_tau = anl._calculate_prediction_error(ground_truth_stim,best_intensity_psth,gt_integral,best_intensity_integral,plot=True,path_to_save=tau_xrmse_path,channel_list=stim_elec_list)                
                

                

                
                tmp_xrmse_path = os.path.join(pred_folder_path,'best_int_xrmse')
                if not os.path.exists(tmp_xrmse_path):
                    os.makedirs(tmp_xrmse_path)


                plt.close()
                sns.heatmap(best_intensity_psth,cmap='coolwarm',xticklabels=np.array(list(range(1,best_intensity_psth.shape[1]+1,1)))*int_time)
                plt.title(f'Normalized ISR - Prediction [a.u.], Time bin = {int_time} ms')
                plt.xlabel('Time bin [ms]')
                plt.ylabel('Channel number')
                plt.savefig(os.path.join(tmp_xrmse_path,f'psth_stim_protocol_{stim}_m_{m}_{chr(945)}_{alpha_val}_intensity_{best_intensity}.pdf'))
                plt.close()

                all_alpha_dict_prediction[alpha_val]={'best_intensity_psth':best_intensity_psth,'best_intensity_psth_std':best_intensity_psth_std,'ch_wise_err':ch_wise_error,'intensity':best_intensity}


                all_m_ch_prediction_dict[m][alpha_val]={'best_intensity_psth':best_intensity_psth,'best_intensity_psth_std':best_intensity_psth_std,'ch_wise_err':ch_wise_error,'intensity':best_intensity}


                channel_prediction_folder = os.path.join(alpha_value_specific_path,'channel_prediction_xrmse')
                os.makedirs(channel_prediction_folder)
                
                for ch in range(best_intensity_psth.shape[0]):
                    
                    ch_id = stim_elec_list[ch]
                    plt.close()
                    plt.style.use('seaborn-v0_8')
                    plt.errorbar(np.round(np.array(list(range(1,best_intensity_psth.shape[1]+1,1)))*int_time,1),best_intensity_psth[ch,:],yerr=best_intensity_psth_std[ch,:],label = 'Prediction',color='red',marker='o',linestyle='--',linewidth=0.5)
                    plt.plot(np.array(list(range(1,best_intensity_psth.shape[1]+1,1)))*int_time,ground_truth_stim[ch,:],label='Ground truth',color='blue',marker='o',linestyle='--',linewidth=0.5)
                    plt.title(f'Channel {ch_id} response ISR, Time bin = {int_time} ms')
                    plt.xlabel(f'Time [ms]')
                    plt.ylabel('ISR [a.u.]')
                    plt.legend()
                    plt.savefig(os.path.join(channel_prediction_folder,f'channel_{ch_id}_prediction.pdf'))
                    plt.close()

                # response roc
                max_amplitude_psth = np.max(best_intensity_psth,axis=1)


                binarized_gt_tmp_2 = copy.deepcopy(binarized_gt)

                binarized_gt_roc_2,max_amplitude_psth,roc_curve_path = fix_roc(binarized_gt_tmp_2,max_amplitude_psth,alpha_value_specific_path,gt_thr=gt_thr)
                
                auc = anl._roc_curve(binarized_gt_roc_2,max_amplitude_psth,f'm={m},{chr(945)}={alpha_val},intensity={best_intensity}',roc_curve_path)

                heatmap_xrmse_xrmse['m'].append(m),heatmap_xrmse_xrmse['alpha'].append(alpha_val),heatmap_xrmse_xrmse['best_intensity'].append(best_intensity),heatmap_xrmse_xrmse['xrmse'].append(overall_error)
                
                heatmap_auc_xrmse['m'].append(m),heatmap_auc_xrmse['alpha'].append(alpha_val),heatmap_auc_xrmse['best_intensity'].append(best_intensity),heatmap_auc_xrmse['auc'].append(auc)

                

                stims_final_xrmse_csv['stim'].append(stim),stims_final_xrmse_csv['m'].append(m),stims_final_xrmse_csv['alpha'].append(alpha_val)
                stims_final_xrmse_csv['intensity'].append(best_intensity),stims_final_xrmse_csv['xrmse'].append(overall_error),stims_final_xrmse_csv['auc'].append(auc),stims_final_xrmse_csv['tau'].append(overall_tau)

                if exp.exp_type == 'exp':
                    fpr, tpr, ths = metrics.roc_curve(binarized_gt_roc_2, max_amplitude_psth)

                    #tmp_fpr = fpr[fpr>0]
                    tmp_tpr = tpr[tpr<1]
                    selected_ths = ths[np.argmax(tmp_tpr)]
                    
                    #selected_ths = ths[((len(fpr)-len(tmp_fpr))+np.argmin(tmp_fpr))]
                    print('ciao')
                    
                    print(ths)
                    print(selected_ths)
                    print(tpr)
                    print(tmp_tpr)
                    
                    
                    print('addio')
            
                    binarized_max_amplitude_psth = np.where(max_amplitude_psth>=selected_ths,1,0)


                    lowest_error = np.min(overall_err_list) # lowest xrmse value reached

                    if lowest_error < best_overall_xrmse_channel_int_prediction:
                        best_overall_xrmse_channel_int_prediction = lowest_error
                        best_channel_int_prediction =max_amplitude_psth # variable where to store the channel integral of the best prediction based on xrmse
                        best_m_channel_int_prediction = m # same but for m
                        best_alpha_channel_int_prediction = alpha_val # same but for alpha
                        best_int_channel_int_prediction = best_intensity # same for intensity
                        ch_wise_error_int_prediction = ch_wise_error
                        best_channel_int_bin_prediction = binarized_max_amplitude_psth # variable where to store the channel integral of the best prediction based on xrmse

               

            #save best alpha for response
            best_alpha_for_protocol_xrmse = math.inf
            alpha_of_best = None
            
            # all alphas plot xrmse vs intensity
            plt.close()
            plt.style.use('seaborn-v0_8')
            color_list = ['blue','orange','green','red','purple','brown','pink','olive','cyan','rebeccapurple','gold']
            for num,alpha_val in enumerate(sorted(list(all_alpha_dict.keys()))):

                #overall_error vs intensity
                plt.plot(intensity_values,all_alpha_dict[alpha_val]['xrmse'],label=f'{chr(945)}={alpha_val}',color=color_list[num])

                best_xrmse_for_alpha = np.min(all_alpha_dict[alpha_val]['xrmse'])
                if best_xrmse_for_alpha<best_alpha_for_protocol_xrmse:
                    best_alpha_for_protocol_xrmse = best_xrmse_for_alpha
                    alpha_of_best = alpha_val

            plt.xlabel('Intensity [a.u.]')
            plt.ylabel('$\\overline{R}$')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(os.path.join(all_alpha_path,f'xrmse_vs_intensity_stim_{stim}_m_{m}_all_alphas.pdf'),bbox_inches='tight')
            plt.close()

            best_intensity_alpha = intensity_values[np.argmin(np.array(all_alpha_dict[alpha_of_best]['xrmse']))]

            
            # all alphas plot auc vs intensity
            plt.close()
            plt.style.use('seaborn-v0_8')
            color_list = ['blue','orange','green','red','purple','brown','pink','olive','cyan','rebeccapurple','gold']
            for num,alpha_val in enumerate(sorted(list(all_alpha_dict.keys()))):

                #overall_error vs intensity
                plt.plot(intensity_values,all_alpha_dict[alpha_val]['auc'],label=f'{chr(945)}={alpha_val}',color=color_list[num])


            
            plt.xlabel('Intensity [a.u.]')
            plt.ylabel('AUC')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(os.path.join(all_alpha_path,f'auc_vs_intensity_stim_{stim}_m_{m}_all_alphas.pdf'),bbox_inches='tight')
            plt.close()

            #alpha all channel prediction
            
            tmp_rearranged ={}
            for alpha in sorted(all_alpha_dict_prediction.keys()):
                    psth = all_alpha_dict_prediction[alpha]['best_intensity_psth']
                    psth_std = all_alpha_dict_prediction[alpha]['best_intensity_psth_std']
                    ch_err = all_alpha_dict_prediction[alpha]['ch_wise_err']
                    best_intensity = all_alpha_dict_prediction[alpha]['intensity']


                    for ch in range(len(ch_err)):
                        if ch not in tmp_rearranged: 
                            tmp_rearranged[ch] = {'pred':[psth[ch]],'pred_std':[psth_std[ch]],'ch_err':[ch_err[ch]],'int':[best_intensity]}
                        else:
                            tmp_rearranged[ch]['pred'].append(psth[ch])
                            tmp_rearranged[ch]['ch_err'].append(ch_err[ch])
                            tmp_rearranged[ch]['int'].append(best_intensity)
                            tmp_rearranged[ch]['pred_std'].append(psth_std[ch])

            
            for ch in sorted(tmp_rearranged.keys()):
                plt.close()
                all_pred_alpha_ch = tmp_rearranged[ch]['pred']
                ch_err_alpha_ch = tmp_rearranged[ch]['ch_err']
                int_alpha_ch = tmp_rearranged[ch]['int']
                pred_std_alpha_ch = tmp_rearranged[ch]['pred_std']
                for alpha_ch in range(len(all_pred_alpha_ch)):
                    plt.style.use('seaborn-v0_8')
                    plt.errorbar(np.array(list(range(1,len(all_pred_alpha_ch[alpha_ch])+1,1)))*int_time,all_pred_alpha_ch[alpha_ch],yerr=pred_std_alpha_ch[alpha_ch],label = f'Prediction: {chr(945)}={alphas_values[alpha_ch]},intensity={round(int_alpha_ch[alpha_ch],1)},{chr(949)}={round(ch_err_alpha_ch[alpha_ch],3)} ',color=color_list[alpha_ch],marker='o',linestyle='--',linewidth=0.5)
                
                plt.plot(np.array(list(range(1,best_intensity_psth.shape[1]+1,1)))*int_time,ground_truth_stim[ch,:],label='Ground truth',color='black',marker='o',linestyle='--',linewidth=0.5)
                
                plt.xlabel(f'Time (ms)')
                plt.ylabel('ISR [a.u.]')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.savefig(os.path.join(all_alpha_path,f'channel_{stim_elec_list[ch]}_m_{m}_all_alpha_prediction.pdf'),bbox_inches='tight')
                plt.close()

        plot_mea_map(unfiltered_elec_list,stim_elec_list,best_channel_int_bin_prediction,stim,'Response integral [a.u.]',os.path.join(stim_channel_int_folder,f'stim_{stim}_binarized_prediction_integral.pdf'))
        plot_mea_map(unfiltered_elec_list,stim_elec_list,best_channel_int_prediction,stim,'Response integral [a.u.]',os.path.join(stim_channel_int_folder,f'stim_{stim}_prediction_integral.pdf'))
                        
        plot_mea_map(unfiltered_elec_list,stim_elec_list,ch_wise_error_int_prediction,stim,'$\\varepsilon$',os.path.join(stim_channel_int_folder,f'stim_{stim}_ch_wise_err_prediction_integral.pdf'))
        
        print('choose value:',best_m_channel_int_prediction,best_alpha_channel_int_prediction,best_int_channel_int_prediction,best_overall_xrmse_channel_int_prediction)
                        



        #heatmapss

        heatmap_xrmse_xrmse = pd.DataFrame(heatmap_xrmse_xrmse)
        heatmap_auc_xrmse = pd.DataFrame(heatmap_auc_xrmse)
        heatmap_xrmse_auc = pd.DataFrame(heatmap_xrmse_auc)
        heatmap_auc_auc = pd.DataFrame(heatmap_auc_auc)


        plt.close()
        sns.heatmap(heatmap_xrmse_xrmse.pivot(index="m", columns="alpha", values="xrmse"),cmap='coolwarm')
        plt.xlabel(f'{chr(945)}')
        plt.title('$\\overline{R}$')
        plt.savefig(os.path.join(single_stim_path,'m_alpha_xrmse_using_xrmse.pdf'))
        
        plt.close()

        sns.heatmap(heatmap_xrmse_xrmse.pivot(index="m", columns="alpha", values="best_intensity"),cmap='coolwarm')
        plt.xlabel(f'{chr(945)}')
        plt.title('Optimal intensity [a.u.]')
        plt.savefig(os.path.join(single_stim_path,'m_alpha_best_intensity_xrmse_using_xrmse.pdf'))
        plt.close()

        sns.heatmap(heatmap_auc_xrmse.pivot(index="m", columns="alpha", values="auc"),cmap='coolwarm')
        plt.xlabel(f'{chr(945)}')
        plt.title('Response Prediction ROC AUC')
        plt.savefig(os.path.join(single_stim_path,'m_alpha_auc_using_xrmse.pdf'))
        plt.close()
        sns.heatmap(heatmap_auc_xrmse.pivot(index="m", columns="alpha", values="best_intensity"),cmap='coolwarm')
        plt.xlabel(f'{chr(945)}')
        plt.title('Optimal intensity [a.u.]')
        plt.savefig(os.path.join(single_stim_path,'m_alpha_best_intensity_auc_using_xrmse.pdf'))

        plt.close()
        sns.heatmap(heatmap_xrmse_auc.pivot(index="m", columns="alpha", values="xrmse"),cmap='coolwarm')
        plt.xlabel(f'{chr(945)}')
        plt.title('$\\overline{R}$')
        plt.savefig(os.path.join(single_stim_path,'m_alpha_xrmse_using_auc.pdf'))
        plt.close()

        sns.heatmap(heatmap_xrmse_auc.pivot(index="m", columns="alpha", values="best_intensity"),cmap='coolwarm')
        plt.xlabel(f'{chr(945)}')
        plt.title('Optimal intensity [a.u.]')
        plt.savefig(os.path.join(single_stim_path,'m_alpha_best_intensity_xrmse_using_auc.pdf'))
        plt.close()

        sns.heatmap(heatmap_auc_auc.pivot(index="m", columns="alpha", values="auc"),cmap='coolwarm')
        plt.xlabel(f'{chr(945)}')
        plt.title('Response Prediction ROC AUC')
        plt.savefig(os.path.join(single_stim_path,'m_alpha_auc_using_auc.pdf'))
        plt.close()

        sns.heatmap(heatmap_auc_auc.pivot(index="m", columns="alpha", values="best_intensity"),cmap='coolwarm')
        plt.title('Optimal intensity [a.u.]')
        plt.xlabel(f'{chr(945)}')
        plt.savefig(os.path.join(single_stim_path,'m_alpha_best_intensity_auc_using_auc.pdf'))
        plt.close()




        # all m plot xrmse vs intensity
        
        tmp_all_m = {}


        for m_val in sorted(list(all_m_dict.keys())):

            for alpha_val in sorted(list(all_m_dict[m_val].keys())):

                if alpha_val not in tmp_all_m:
                    tmp_all_m[alpha_val] = {}

                if m_val not in tmp_all_m[alpha_val]:
                    tmp_all_m[alpha_val][m_val]={'xrmse':all_m_dict[m_val][alpha_val]['xrmse'],'auc':all_m_dict[m_val][alpha_val]['auc']}
                




                    
        plt.close()
        plt.style.use('seaborn-v0_8')
        color_list = ['blue','orange','green','red','purple','brown','pink','olive','cyan','rebeccapurple','gold']

        for alpha_val in sorted(list(tmp_all_m.keys())):

            for num,m_val in enumerate(sorted(list(tmp_all_m[alpha_val].keys()))):

                #overall_error vs intensity
                plt.plot(intensity_values,tmp_all_m[alpha_val][m_val]['xrmse'],label=f'm={m_val}',color=color_list[num])


            
            plt.xlabel('Intensity [a.u.]')
            plt.ylabel('$\\overline{R}$')
            plt.legend()
            plt.savefig(os.path.join(single_stim_path,f'xrmse_vs_intensity_stim_{stim}_all_m_{chr(945)}_{alpha_val}.pdf'))
            plt.close()


        # all m plot auc vs intensity
        plt.close()
        plt.style.use('seaborn-v0_8')
        color_list = ['blue','orange','green','red','purple','brown','pink','olive','cyan','rebeccapurple','gold']

        for alpha_val in sorted(list(tmp_all_m.keys())):

            for num,m_val in enumerate(sorted(list(tmp_all_m[alpha_val].keys()))):

                #overall_error vs intensity
                plt.plot(intensity_values,tmp_all_m[alpha_val][m_val]['auc'],label=f'm={m_val}',color=color_list[num])


            
            plt.xlabel('Intensity [a.u.]')
            plt.ylabel('AUC')
            plt.legend()
            plt.savefig(os.path.join(single_stim_path,f'AUC_vs_intensity_stim_{stim}_all_m_{chr(945)}_{alpha_val}.pdf'))
            plt.close()
        


        #m all channel prediction
        color_list = ['blue','orange','green','red','purple','brown','pink','olive','cyan','rebeccapurple','gold']
        tmp_rearranged ={}

        all_m_path = os.path.join(single_stim_path,'all_m')
        os.makedirs(all_m_path)


        all_m_ch_pred_tmp={}

        for m in sorted(list(all_m_ch_prediction_dict.keys())):
                
                for alpha in sorted(list(all_m_ch_prediction_dict[m].keys())):
                    
                    if alpha not in all_m_ch_pred_tmp:
                        all_m_ch_pred_tmp[alpha]={}

                    psth = all_m_ch_prediction_dict[m][alpha]['best_intensity_psth']
                    psth_std = all_m_ch_prediction_dict[m][alpha]['best_intensity_psth_std']
                    ch_err = all_m_ch_prediction_dict[m][alpha]['ch_wise_err']
                    best_intensity = all_m_ch_prediction_dict[m][alpha]['intensity']


                    for ch in range(len(ch_err)):
                        if ch not in all_m_ch_pred_tmp[alpha]: 
                            all_m_ch_pred_tmp[alpha][ch] = {'pred':[psth[ch]],'pred_std':[psth_std[ch]],'ch_err':[ch_err[ch]],'int':[best_intensity]}
                        else:
                            all_m_ch_pred_tmp[alpha][ch]['pred'].append(psth[ch])
                            all_m_ch_pred_tmp[alpha][ch]['ch_err'].append(ch_err[ch])
                            all_m_ch_pred_tmp[alpha][ch]['int'].append(best_intensity)
                            all_m_ch_pred_tmp[alpha][ch]['pred_std'].append(psth_std[ch])

        
        for alpha in sorted(list(all_m_ch_pred_tmp.keys())):
            
            tmp_alpha_path = os.path.join(all_m_path,f'alpha_{alpha}')
            if not os.path.exists(tmp_alpha_path):
                os.makedirs(tmp_alpha_path)

            for ch in sorted(list(all_m_ch_pred_tmp[alpha].keys())):
                plt.close()
                all_pred_m_ch = all_m_ch_pred_tmp[alpha][ch]['pred']
                ch_err_m_ch = all_m_ch_pred_tmp[alpha][ch]['ch_err']
                int_m_ch = all_m_ch_pred_tmp[alpha][ch]['int']
                pred_std_m_ch = all_m_ch_pred_tmp[alpha][ch]['pred_std']
                for m_ch in range(len(all_pred_m_ch)):
                    plt.style.use('seaborn-v0_8')
                    plt.errorbar(np.round(np.array(list(range(1,len(all_pred_m_ch[m_ch])+1,1)))*int_time,1),all_pred_m_ch[m_ch],yerr=pred_std_m_ch[m_ch],label = f'Prediction: m={m_values[m_ch]},intensity={round(int_m_ch[m_ch],1)},{chr(949)}={round(ch_err_m_ch[m_ch],3)} ',color=color_list[m_ch],marker='o',linestyle='--',linewidth=0.5)
                
                plt.plot(np.round(np.array(list(range(1,best_intensity_psth.shape[1]+1,1)))*int_time,1),ground_truth_stim[ch,:],label='Ground truth',color='black',marker='o',linestyle='--',linewidth=0.5)
                
                
                plt.xlabel(f'Time [ms]')
                plt.ylabel('ISR [a.u.]')
                plt.legend()
                plt.savefig(os.path.join(tmp_alpha_path,f'channel_{stim_elec_list[ch]}_all_m_{chr(945)}_{alpha}_prediction.pdf'))
                plt.close()
    
        
    #save all stimulation data
    stims_final_xrmse_csv = pd.DataFrame(stims_final_xrmse_csv)
    stims_final_xrmse_csv.to_csv(os.path.join(pred_folder_path,'stimulus_prediction_all_data_xrmse.csv'),sep='\t')
    stims_final_auc_csv = pd.DataFrame(stims_final_auc_csv)
    stims_final_auc_csv.to_csv(os.path.join(pred_folder_path,'stimulus_prediction_all_data_auc.csv'),sep='\t')


    
    # create folder to store prediction data of all mearc models
    # pred_folder_path = os.path.join(path_to_exp_folder,'all_mearc_cm_analysis')
    # if not os.path.exists(pred_folder_path):
    #      os.makedirs(pred_folder_path)

    # exp.results_path = pred_folder_path

    # anl = analyzer(exp)

    # print(anl.exp.results_path)

    # name_exp = path_to_exp_folder.split('/')[-1]
    # anl.connectivity_matrix_analysis(all_mearc_model,exp_type=exp.exp_type,exp_name=name_exp)
    

    
