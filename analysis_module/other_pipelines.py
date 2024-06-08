
#%%
from data_prep.exp import experiment
from data_prep.analyzer import analyzer
from mea_reservoir.lasso_node import lasso
from mea_reservoir.mea_node import mea_res
from data_prep.utils import save_pickle_file,adjust_font_dimension,open_pickle_file,check_if_file_exist,plot_channel_prediction,plot_single_exp_cm_analysis
import os
import pickle
import numpy as np
from time import time
import pandas as pd
import shutil




def just_stimulation_pipeline(already_analyzed_exp_path,overwrite=False):

    '''
    This pipeline is used to perform just the stimulation analysis. It requires that the experiment of interest has been previuosly analyzed (data processing + meaRC model training).
    
    Input:
        - already_analyzed_exp_path: path leading to the folder obtained as results from performing data processing and training of the meaRC models.
        - overwrite: if set to True, if stimulus_prediction data are already present, they will be overwritten
    '''

    path_exp_pickle = os.path.join(already_analyzed_exp_path,'analyzed_experiment.pickle') # if the data have been already analyzed, we retrieve them from the pickle file 


    check_if_file_exist(path_exp_pickle,f'{path_exp_pickle} does not exist. {already_analyzed_exp_path.name} has not been previously analyzed. Necessary meaRC model training before stimulation analysis.')
    
    exp = open_pickle_file(path_exp_pickle) # loading exp pickle file

    exp.results_path = already_analyzed_exp_path

    anl = analyzer(exp) # analyzer class istance

    #check if training data has been previously created

    training_check = exp.get_data('training') # if the key 'training' is not present in exp data dictionary, no data processing has been performed and error will be raised

    # check that training of meaRC models has been previuosly performed
    path_mearc_models = os.path.join(already_analyzed_exp_path,'characterized_mearc.pickle') # path to the saved meaRC trained models

    check_if_file_exist(path_mearc_models,f'{path_mearc_models} does not exist. No meaRC model training has been performed for {already_analyzed_exp_path.name}.')

    mearc_characterized_dict = open_pickle_file(path_mearc_models)

    # check if grounf truth stimulation data (PSTH) have been calculated
        
    if exp.exp_type =='exp':
        try:
            analyzed_stim_dict =exp.get_data('processed_filt_stimulation')
        except:
            anl.calculate_psth(20,'filt_stimulation')
            
    else:
        try:
            analyzed_stim_dict =exp.get_data('processed_stimulation')
        except:
            anl.calculate_psth(20)
            

    #check if stimulation of meaRC model has been performed. if not it is calculated
    # try:
    #     predicted_stim = exp.get_data('predicted_psth')

    # except:
    #     anl.stimulate_mearc(mearc_characterized_dict,analyzed_stim_dict,20)
    #     predicted_stim = exp.get_data('predicted_psth')

    del exp.data['predicted_psth']

    anl.stimulate_mearc(mearc_characterized_dict,analyzed_stim_dict,20)
    predicted_stim = exp.get_data('predicted_psth')

    # perform full stimulation characterization

    anl.full_characterization_stim(predicted_stim,analyzed_stim_dict,overwrite=overwrite)

    exp.results_path = ''

    os.remove(path_exp_pickle)

    save_pickle_file(exp,path_exp_pickle)

    

            


def run_for_multiple_stim_only(already_analyzed_exp_folder):

    exp_folder = [fold for fold in os.scandir(already_analyzed_exp_folder) if fold.is_dir()]

    for exp in exp_folder:

        just_stimulation_pipeline(exp,overwrite=True)

            

def debug(data_path,path):

    mearc = open_pickle_file(os.path.join(path,'characterized_mearc.pickle'))

    exp=experiment(path=data_path,results_path='',seed=1)

    anl= analyzer(exp)

    anl.calculate_psth(20)



#debug('/home/penelope/Desktop/ilya/mearc_model/final_sim_to_run/60_sims/60_pop_stim_full_pto_2','/home/penelope/Desktop/ilya/mearc_model/all_sim_results/60_pop_stim_full_pto_2')


