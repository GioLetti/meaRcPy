#%%
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Union
#%%

def squared_error(ground_truth:np.ndarray,
                       predicted:np.ndarray):
    
    difference = ground_truth - predicted
    squared_difference = difference**2

    return squared_difference

def absolute_percentage_error(ground_truth:np.ndarray,
                              predicted:np.ndarray):
    
    return np.abs((ground_truth-predicted)/ground_truth)


def cross_metrics(ground_truth:np.ndarray, # ground truth data
                  predicted:np.ndarray, # predicted data
                  channel_integral:np.ndarray, #channel integral vector that is going to be used for weighting the error
                  pred_integral:np.ndarray,
                metric:callable = squared_error, # the function to calculate the desired metric. Default mean_absolute_error of sklearn
                tau_max:int = 10, # maximal time shift for which the metric is calculated. tested taus go from -tau_max to +tau_max
                plot:bool = False,
                path_to_save:str = None,
                int_time:Union[float,int]=None,
                channel_list:list=None):
    '''
    Input:
      - ground_truth: (n_samples x n_outputs) as in sklearn
      - predicted: (n_samples x n_outputs) as in sklearn
    
    '''
    
    tested_tau = np.arange(-tau_max,tau_max+1,1) # tau value for which the metric is calculated

    error_ = []

    max_integral = np.maximum(channel_integral,pred_integral)

    #channel_integral_norm = channel_integral/np.sum(channel_integral)

    max_integral_norm = max_integral/np.sum(max_integral)

    ch_number = ground_truth.shape[0]
    ground_truth_with_added_zeros = np.concatenate((np.zeros((ch_number,tau_max)),ground_truth,np.zeros((ch_number,tau_max))),axis=1) # in this way we can shift the prediction over the response

    for tau in tested_tau:
        
        shifted_pred = np.concatenate((np.zeros((ch_number,tau_max+tau)),predicted,np.zeros((ch_number,tau_max-tau))),axis=1)
        tmp = np.sum((ground_truth_with_added_zeros + shifted_pred),axis=1)
        weight_component = np.divide((ground_truth_with_added_zeros + shifted_pred),tmp[:,np.newaxis]) # weight calculation
        err = metric(ground_truth_with_added_zeros,shifted_pred) # metric error

        weighted_err = np.sqrt(np.sum(weight_component * err,axis=1)) # RMSE

        error_.append(weighted_err)

    
        
    error_ = np.squeeze(np.dstack(error_))

    min_err_channel_wise = np.min(error_,axis=1)
    min_err_tau_channel_wise = tested_tau[np.argmin(error_,axis=1)]

    ch_error = min_err_channel_wise* max_integral_norm
 

    ch_tau =  min_err_tau_channel_wise * max_integral_norm


    if plot == True:
        
        for ch in range(error_.shape[0]):
            
            ch_id = channel_list[ch]

            plt.close()
            plt.plot(tested_tau*int_time,error_[ch,:],marker='o',linestyle='--',color='blue',linewidth=0.5)
            plt.title(f'channel {ch_id}')
            plt.xlabel(f'{chr(964)} [ms]')
            plt.ylabel('XRMSE')
            plt.savefig(os.path.join(path_to_save,f'channel_{ch_id}_xrmse_vs_{chr(964)}.pdf'),bbox_inches='tight')
            plt.close()

        
      
    overall_error = np.sum(ch_error)
    overall_tau = np.sum(ch_tau)

    return min_err_channel_wise,min_err_tau_channel_wise,overall_error,overall_tau


    
        

        




        
