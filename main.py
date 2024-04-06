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

#INPUT_PATH = '/home/penelope/Desktop/ilya/mearc_model/exp_files_final/Final_output'
#OUTPUT_PATH='/home/penelope/Desktop/ilya/mearc_model/all_exp_results/pt2'

#INPUT_PATH = '/home/penelope/Desktop/ilya/mearc_model/all_sim_results'

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

    

            



def run_for_multiple_exp(input_folder_path,output_folder_path):
    
    
    exp_folder = [fold for fold in os.scandir(input_folder_path) if fold.is_dir()]

    already_analyzed_fold = [fold.name for fold in os.scandir(output_folder_path) if fold.is_dir()]

    random_seed = int(np.random.randint(1,100000,1)[0])
    
    for exp in exp_folder:

        if exp.name not in already_analyzed_fold:

            analysis_pipeline_single_exp(exp,output_folder_path,seed=random_seed)


def run_for_multiple_stim_only(already_analyzed_exp_folder):

    exp_folder = [fold for fold in os.scandir(already_analyzed_exp_folder) if fold.is_dir()]

    for exp in exp_folder:

        just_stimulation_pipeline(exp,overwrite=True)

            

def debug(data_path,path):

    mearc = open_pickle_file(os.path.join(path,'characterized_mearc.pickle'))

    exp=experiment(path=data_path,results_path='',seed=1)

    anl= analyzer(exp)

    anl.calculate_psth(20)




# if __name__=='__main__':

    
#     if not os.path.exists(INPUT_PATH):
#         raise IOError(f'{INPUT_PATH} does not exist')

#     if not os.path.exists(OUTPUT_PATH):
#         os.makedirs(OUTPUT_PATH)

#     adjust_font_dimension()
    
#     run_for_multiple_exp(INPUT_PATH,OUTPUT_PATH)
#     #run_for_multiple_stim_only(INPUT_PATH)



#debug('/home/penelope/Desktop/ilya/mearc_model/final_sim_to_run/60_sims/60_pop_stim_full_pto_2','/home/penelope/Desktop/ilya/mearc_model/all_sim_results/60_pop_stim_full_pto_2')


#%%
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%%
fig,axs = plt.subplots(2,2)
axs = axs.flatten()
axs[2].secondary_yaxis(-0.5,ylim=10)
axs[2].secondary_xaxis(-0.5,xlim=10)
#axins.tick_params(left=False, right=True, labelleft=False, labelright=True)
plt.show()

# %%
#cm = open_pickle_file('/home/giorgio/Desktop/fin_11_01/cm_analysis_results.pickle')
# %%
adjust_font_dimension(title_size=20,legend_size=25,label_size=18,x_ticks=15,y_ticks=15)

exp = open_pickle_file('/home/giorgio/Desktop/fin_11_01/60_pop_stim_full_inh_isol__pto_1_results/analyzed_experiment.pickle') # loading exp pickle file

exp.results_path = '/home/giorgio/Desktop/fin_11_01/60_pop_stim_full_inh_isol__pto_1_results/final'
#exp.exp_path = '/home/giorgio/Desktop/60_pop_stim_full_inh__isol_pto_1'

anl = analyzer(exp)

mearc_characterized_dict = open_pickle_file('/home/giorgio/Desktop/fin_11_01/60_pop_stim_full_inh_isol__pto_1_results/characterized_mearc.pickle')

predicted_stim = exp.get_data('predicted_psth')

analyzed_stim_dict =exp.get_data('processed_stimulation')

#anl.connectivity_matrix_analysis(mearc_characterized_dict,exp_type=exp.exp_type)

anl.full_characterization_stim(predicted_stim,analyzed_stim_dict,overwrite=False)
# %%

a = '/home/giorgio/Desktop/fin_11_01'
adjust_font_dimension(title_size=30,legend_size=60,label_size=40,x_ticks=20,y_ticks=18)

plot_channel_prediction(os.path.join(a,'40628_26DIV','analyzed_experiment.pickle'),13,50,0.2,2.5,a+'/last/last/exp',sim=False)

# %%
a = '/home/giorgio/Desktop/fin_11_01/60_pop_stim_full_inh_isol__pto_1_results'
adjust_font_dimension(title_size=20,legend_size=25,label_size=18,x_ticks=15,y_ticks=15)
exp = open_pickle_file(os.path.join(a,'analyzed_experiment.pickle'))
# exp.results_path = a+'/val'
# mea = open_pickle_file(os.path.join(a,'characterized_mearc.pickle'))

# anl = analyzer(exp)

# anl._find_best_mearc(mea)

# %% ISI And IBI plot
adjust_font_dimension(title_size=20,legend_size=25,label_size=18,x_ticks=15,y_ticks=15)

exp = open_pickle_file('/home/giorgio/Desktop/fin_11_01/60_pop_stim_full_inh_isol__pto_1_results/analyzed_experiment.pickle')
exp.results_path = '/home/giorgio/Desktop/fin_11_01/60_pop_stim_full_inh_isol__pto_1_results/final'

anl = analyzer(exp)
anl.calculate_isith(anl.retrieve_data_exp('background')[0] ,just_plot=True)

burst_data = anl.retrieve_data_exp('burst') # retrieve burst data

anl.calculate_ibith(burst_data,just_plot=True)
#%% RASTER AND SINGLE POPULATION ISI PLOT
#adjust_font_dimension(title_size=20,legend_size=25,label_size=18,x_ticks=15,y_ticks=15)
adjust_font_dimension(title_size=20,legend_size=25,label_size=18,x_ticks=15,y_ticks=15)

exp = open_pickle_file('/home/giorgio/Desktop/fin_11_01/60_pop_stim_full_inh_isol__pto_1_results/analyzed_experiment.pickle')
exp.results_path = '/home/giorgio/Desktop/fin_11_01/60_pop_stim_full_inh_isol__pto_1_results/final'


anl = analyzer(exp)
anl.raster_plot(start=100000,stop=140000,filt=False)
anl.plot_single_population_isi(36)


# %%PLOT CM ANALYSIS
adjust_font_dimension(title_size=20,legend_size=25,label_size=18,x_ticks=15,y_ticks=15)

plot_single_exp_cm_analysis('/home/giorgio/Desktop/fin_11_01/60_pop_stim_full_inh_isol__pto_1_results/cm_analysis_results.pickle','/home/giorgio/Desktop/fin_11_01/60_pop_stim_full_inh_isol__pto_1_results/final/cm_analysis')

# %%


#%%
from scipy.io import savemat
#%%
cms = open_pickle_file('/home/giorgio/Desktop/fin_11_01/40628_26DIV/cm_analysis/cm_analysis_results.pickle') # loading exp pickle file

savemat(os.path.join('/home/giorgio/Desktop/fin_11_01/40628_26DIV','cm_50_02.mat'),cms[50][0.2])
# %%
