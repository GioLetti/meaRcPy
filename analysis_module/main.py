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

#%%



INPUT_PATH = '/home/penelope/Desktop/ilya/mearc_model/final_sim_to_run/final_sim'
OUTPUT_PATH='/home/penelope/Desktop/ilya/mearc_model/all_sim_results'




def analysis_pipeline_single_exp(exp_path,exp_output_path,seed:int=1):
    
    #The function performs full analysis of experiment, both experimental or simulations
    

    start_prepocessing = time()
    exp=experiment(path=exp_path,results_path=exp_output_path,seed=seed)

    exp_seed = exp.seed

    anl = analyzer(exp)

    if exp.exp_type=='exp':
        anl.filter_spikes_data('background',0.99)
        anl.filter_stim_data(anl.retrieve_data_exp('stimulation'),anl.retrieve_data_exp('filt_background')[1])
        try:
            anl.calculate_isith(anl.retrieve_data_exp('filt_background')[0])
        except:
            print('No ISITh computable, skipping analysis')
            return 

        

        anl.calculate_burst(anl.retrieve_data_exp('filt_background')[0],anl.retrieve_data_exp('filt_background')[1],0)
        

    else:
    # analysis performing burst and network burst detection, followed by creation of training data
        anl.calculate_isith(anl.retrieve_data_exp('background')[0])
        anl.calculate_burst(anl.retrieve_data_exp('background')[0],anl.retrieve_data_exp('background')[1],0)
    
    burst_data = anl.retrieve_data_exp('burst') # retrieve burst data

    try:
        anl.calculate_ibith(burst_data)
    except:
        print('No IBITh computable, skipping analysis')
        return

    anl.calculate_network_burst(burst_data)
    if exp.exp_type == 'exp':
        anl.create_training_data(0.85,'filt_background',extra_steps=50)
    else:
        anl.create_training_data(0.85,extra_steps=50)

    # prepering training and validation data
    norm_factor = exp.get_param('norm_factor_back')
    training = exp.get_data('training')
    training = [tr/norm_factor for tr in training]

    validation = exp.get_data('validation')
    validation = [val/norm_factor for val in validation]

    end_preprocessing = time()
    preprocessing_time = end_preprocessing-start_prepocessing


    # run meaRC charcaterization
    start_char = time()


    # TO RUN WITH BIAS USE THIS
    #mearc_characterized_dict = anl.characterize_meaRC(training=training,validation=validation,alphas=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],ms=[1,5,10,20],repeat=5,n_jobs=-1,fit_intercept=True,save_data=True,rng=exp_seed)

    # TO RUN WITHOUT BIAS USE THIS
    mearc_characterized_dict = anl.characterize_meaRC(training=training,validation=validation,alphas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],ms=[1,2,5,10,25],repeat=5,n_jobs=-1,fit_intercept=False,save_data=True,rng=exp_seed)

    #mearc_characterized_dict = anl.characterize_meaRC(training=training,validation=validation,alphas=[0,0.1],ms=[1,2],repeat=2,n_jobs=10,fit_intercept=False,save_data=True,rng=exp_seed)
    end_char = time()
    time_char = end_char-start_char
   
    # calculate psth
    if exp.exp_type == 'exp':
        anl.calculate_psth(20,'filt_stimulation')
    else:
        anl.calculate_psth(20)

    # connectivity matrix
    start_cm = time()
    anl.connectivity_matrix_analysis(mearc_characterized_dict,exp_type=exp.exp_type)
    end_cm = time()
    time_cm = end_cm-start_cm


    stim_start = time()
    if exp.exp_type =='exp':
        analyzed_stim_dict =exp.get_data('processed_filt_stimulation')
    
    else:
        analyzed_stim_dict =exp.get_data('processed_stimulation')

    anl.stimulate_mearc(mearc_characterized_dict,analyzed_stim_dict,20)

    predicted_stim = exp.get_data('predicted_psth')

    anl.full_characterization_stim(predicted_stim,analyzed_stim_dict)

    stim_end = time()
    stim_time = stim_end - stim_start

    time_df = pd.DataFrame({'preprocessing':[preprocessing_time],'mearc_characterization':[time_char],'cm':[time_cm],'stimulation':[stim_time],'overall':[preprocessing_time+time_char+stim_time+time_cm]})
    time_df.to_csv(os.path.join(exp.results_path,'computational_time.csv'),'\t')
    # save experiment
    exp.exp_path=''
    tmp_path = exp.results_path
    exp.results_path=''
    save_pickle_file(exp,os.path.join(tmp_path,'analyzed_experiment.pickle'))



def run_for_multiple_exp(input_folder_path,output_folder_path):
    
    
    exp_folder = [fold for fold in os.scandir(input_folder_path) if fold.is_dir()]

    already_analyzed_fold = [fold.name for fold in os.scandir(output_folder_path) if fold.is_dir()]

    random_seed = int(np.random.randint(1,100000,1)[0])
    
    for exp in exp_folder:

        if exp.name not in already_analyzed_fold:

            analysis_pipeline_single_exp(exp,output_folder_path,seed=random_seed)



if __name__=='__main__':

    
    if not os.path.exists(INPUT_PATH):
        raise IOError(f'{INPUT_PATH} does not exist')

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    adjust_font_dimension()
    
    run_for_multiple_exp(INPUT_PATH,OUTPUT_PATH)
    



