import pandas as pd
import numpy as np
import os
import re
import math
from typing import Union
from pathlib import Path
from functools import partial

# functions used in the __init__ of experiment class

def _get_ch_label(file_name):

    return re.findall('\d\d*',file_name)[0]

def _read_meta(path: os.path):
    '''
    The function extract necessary metadata from the meta_data.csv stored in the metadata folder

    Input:
        - path: path to the metadata folder of an experiment/simulation folder
    Output:
        - recording_duration: the recording duration in second
        - frequency: the sampling freqeuncy expressed in Hz
        - stim_type: if any, it specifies the type of stimulation
    '''

    file_path = os.path.join(path,'meta_data.csv')

    if not os.path.exists(file_path):

        raise FileNotFoundError (f'Path: {file_path} does not exist.')
    
    else:
        tmp=pd.read_csv(file_path,sep='\t')

        recording_duration = tmp['recording_duration_sec'].values # recording duration in seconds
        frequency = tmp['sampling_fr_hz'].values # sampling frequency in Hz

        try:
            tmp['stimulation'].values # stimulation type
        except:
            stim_type = None
        else:
            stim_type = tmp['stimulation'].values

        return recording_duration,frequency,stim_type
    
def _read_stimulation(path):
    '''
    The function retrieve the starting time of the first (if any) stimulation protocol. This allow to separate between spontaneous and induced activity.
    Moreover it creates a dictionary containing key:value == channel_label:stimulation_dataframe of all stimulation protocol
    '''
    stim_file = [file for file in os.scandir(path) if file.is_file()] # retrieve all stimulation file (.csv)

    if len(stim_file)<1: 
        raise ValueError('stimulation files are not present.')
    else:
        stim_dict={} # each key:value will be ch_label:stimulation_dataframe
        starting_time_stim = math.inf 

        for st_f in stim_file:
            
            tmp = pd.read_csv(st_f.path,sep='\t')
            if len(tmp.columns)<=1:
                tmp = pd.read_csv(st_f.path,sep=',')
            

            ch_label = tmp['target'].values[0]
            start = float(tmp['start'].values[0])

            if start < starting_time_stim: # we retrieve the starting time of the stimulations, defined as the time of the first stimuli of the first stimulation protocol
                starting_time_stim=start

            stim_dict[ch_label]=tmp

        return stim_dict,starting_time_stim


def _check_exp_type(path):
    '''
    The function is used to understand the type of experiment (experimental recording or simulation) that is being analyzed.
    It checks the presence of the log_file.log which is present only in the simulation folder
    '''
    if os.path.exists(os.path.join(path,'log_file.log')):
        return 'sim'
    else:
        return 'exp'

def _load_sim_or_exp(path):

    try:
        return np.genfromtxt(path,delimiter='\t',skip_header=1,usecols=1) # in case of simulation this work
    except:
        return np.genfromtxt(path,delimiter='\t',skip_header=1) # instead, in case of experimental recording this work
    else:
        return np.genfromtxt(path,delimiter='\t',skip_header=1,usecols=1) 



def _create_array_from_spikes(path: os.path) -> [np.ndarray,list]:
    '''
    The function combines the spikes data (each electrode has a tsv file) in a np.array of dimension (n_channel x fin_dim)
    where fin_dim is the highest number of spikes detected among all electrodes. Each entry of the tsv file is the spike timestamp.
    Multiplying this value by the sampling frequency (in Hz) gives the spike time in ms
    '''

    spike_files = [sf for sf in os.scandir(path) if (sf.is_file and sf.name.startswith('electrode'))] # electrode csv files containing spiking timestamp

    fin_dim = 0 # columns number of the final data matrix
    # the first for loop is used to identify the dimension
    for sf in spike_files:

        tmp = _load_sim_or_exp(sf.path)
        try: # this avoid problem when there are no spikes or only one is present
            len(tmp)
        except:
            pass
        else:
            if len(tmp)>fin_dim:
                fin_dim=len(tmp)

    

    fin_mat=np.zeros((len(spike_files),fin_dim)) # the final data matrix is initialized. It has dimension channel_number x spikes timestamp

    elec_list=[] # the list containing the electrode order
    for num,sf in enumerate(sorted(spike_files,key=lambda x: x.name)):

        ch_label=_get_ch_label(sf.name).lstrip('0')
        if ch_label=='': # as the first electrode has number 000, the lstrip leaves ''
            ch_label='0'
        elec_list.append(ch_label)
        tmp = _load_sim_or_exp(sf.path)
        try:
            len(tmp)
        except:
            fin_mat[num,0]=tmp
        else:
            fin_mat[num,:len(tmp)]=tmp

    return fin_mat,elec_list



def _read_exp_recording(path:os.path) -> dict:

    '''
    The function creates the data dictionary of an experiment set of an experimental recording session, which is constitued 
    by background + stimulation phases. For the background activity and each stimulation protocol(s) a different folder is present.

    Input:
        - path: the path to the folder containing the analyzed data of one experimental set
    Output:
        - data: the data dictionary
    '''

    exp_folder = [exp_f for exp_f in os.scandir(path) if exp_f.is_dir]

    data_dict = {} # dictionary containing the data of background and stimulation phases {'background':[np.ndarray,list],'stimulation':{channel_label:[pd.DataFrame,np.ndarray,list]}}

    frequency = 0 # sampling frequency

    for exp_f in exp_folder: # for each experimental folder (which can be background or stimulation)
        
        if 'stim' not in exp_f.name.lower():
            recording_duration,frequency_tmp,_ = _read_meta(os.path.join(exp_f.path,'metadata')) # retreive sampling frequency
        else:
            _,frequency_tmp,_ = _read_meta(os.path.join(exp_f.path,'metadata')) # retreive sampling frequency
        # check of sampling frequency that must be equal considering background and stimulation of the same experiment
        if frequency==0:
            frequency=frequency_tmp
        elif frequency_tmp != frequency:
            raise ValueError ('Different sampling frequency have been detected in the same experiment. Same experiment (background+stimulation) must have same sampling frequency')

        if 'stim' not in exp_f.name.lower(): # so data of the background activity
            
            background,elec_order = _create_array_from_spikes(exp_f.path) # the data array is created. Furthermore the list of electrode order is retrieved
            background = np.where(background!=0,background,np.nan)
            if 'background' not in data_dict:
                data_dict['background']=[background,elec_order]
            else:
                raise ValueError(f'Two background files have been found in folder {path}. Only one is expected')

        else:
            if 'stimulation' not in data_dict:
                data_dict['stimulation']={}

            stim_folder_path = os.path.join(exp_f.path,'stimulation_protocols')
            stim_dict,_ =_read_stimulation(stim_folder_path)
            tmp_stim_data,tmp_stim_elec_list = _create_array_from_spikes(exp_f.path)
            tmp_stim_data = np.where(tmp_stim_data!=0,tmp_stim_data,np.nan)
            for key in stim_dict:
                
                if key not in data_dict['stimulation']:

                    data_dict['stimulation'][key]=[stim_dict[key],tmp_stim_data,tmp_stim_elec_list]
    

    return data_dict,frequency,recording_duration

def _check_sim_stimulation(stimulation_path:os.path):
    '''
    The function checks if stimulation protocols have been used in the current simulation experiment
    '''
    return os.path.exists(stimulation_path)




def _read_simulation(path:os.path) -> dict:
    '''
    The function creates the data dictionary for a simulation set, which is constitued by background+stimulation, but all stored in a single folder.
    '''
    data_folder_path = os.path.join(path,'data')


    stim_folder_path= os.path.join(data_folder_path,'stimulation_protocols')

    stim_yes_no = _check_sim_stimulation(stim_folder_path) # returns True if stimulation protocols have been detected, else False

    data_dict={}

    recording_duration,frequency,_ = _read_meta(os.path.join(path,'metadata')) # frequency in Hz

    sampling_period = (1/frequency)*1000 # sampling period in milli second

    if stim_yes_no:
        stim_dict,starting_time_stim = _read_stimulation(stim_folder_path)

    tmp_data,elec_list = _create_array_from_spikes(data_folder_path) # all data, both background and stimulations

    if stim_yes_no:
        background= np.where(((tmp_data!=0) & ((tmp_data*sampling_period)<starting_time_stim)),tmp_data,np.nan) # background activity only
    else:
        background = np.where(tmp_data!=0,tmp_data,np.nan)

    # divide in background and stimulation
    if 'background' not in data_dict:
        data_dict['background']=[background,elec_list]
    if stim_yes_no:
        if 'stimulation' not in data_dict:
            data_dict['stimulation']={}

        for elec in stim_dict:

            start = stim_dict[elec]['start'].values[0]
            end = stim_dict[elec]['end'].values[-1]


            stim_data_tmp = np.where((((tmp_data*sampling_period)>=start) & ((tmp_data*sampling_period)<=end)),tmp_data,np.nan)

            if elec not in data_dict['stimulation']:
                data_dict['stimulation'][elec]=[stim_dict[elec],stim_data_tmp,elec_list]
                

    
    
    return data_dict,frequency,recording_duration 





class experiment():

    def __init__(self,path:Path,results_path:Path,seed:Union[None,int]=None):
        
        #check if the experiment is an experimental recording or simulation
        self.exp_type = _check_exp_type(path)

        #path to the folder of experimental/simulation data
        self.exp_path = Path(path)

        #create results path
        
        exp_name = self.exp_path.name
        

        self.results_path = os.path.join(results_path,exp_name)

        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        #else:
        #    raise FileExistsError(f'Folder {self.results_path} already exist.')
           
            
        #data dictionary
        self.data={}
        #params dictionary
        self.params={}

        #create random number generator
        if seed == None: # if seed is not specified, 1 will be used to avoid problem with scipy
            self.seed,self.rng = seed,np.random.default_rng(1)
        else:
            self.seed,self.rng = seed,np.random.default_rng(seed)

        #based on the exp_type a different function for retrieving data is used
        if self.exp_type == 'exp':
            
            self.data,self.params['sampling_fr'],self.params['duration_in_sec'] =_read_exp_recording(path)

        elif self.exp_type == 'sim':
            self.data,self.params['sampling_fr'],self.params['duration_in_sec'] = _read_simulation(path) 
        

        #check the correct parsing of the data
        
        if self.data.get('background') == None:
           raise ValueError('No background activity has been detected. Analysis can not be performed. Exiting.')
        
        if self.data.get('stimulation')== None:
            print('No stimulation activity has been detected. Analysis will be performed only on background activity.')

        if self.params.get('sampling_fr') == None:
            raise ValueError('No sampling frequency identified. Analysis can not be performed.')

    def __check_data_before_training(self):
        '''
        The function checks that all the data and parameters necessary to create the training/validation data are stored in the experiment istance
        '''

        self.get_data('filt_background') # check that background activity is present in data dictionary

        self.get_data('network_burst') # check that network bursts have been calculated

        self.get_param('int_time') # check that the integration time has been identified

        #self.get_param('batch_time') # check that the batch time has been identified

        print('All necessary data and parameters are present. Trainig/validation data can be created.')


    def set_data(self,name:str,data_to_add):
        '''
        the function add to the data dictionary the couple name:data
        '''

        if name not in self.data:
            self.data[name]=data_to_add
        else:
            raise ValueError(f'{name} already present in the data dictionary')
    
    def get_data(self,name:str):
        '''
        The function retrieves the desired data from the data dictionary
        '''
        dd = self.data.get(name) 
        if isinstance(dd, type(None)):
            raise ValueError (f'{name} is not present in the data dictionary')
        else:
            return dd

    def set_param(self,name:str,param_to_add):
        
        if name not in self.params:
            self.params[name]=param_to_add
        else:
            raise ValueError(f'{name} already present in the data dictionary')
        
    def get_param(self,name):
        
        par = self.params.get(name)
        if isinstance(par,type(None)):

            raise ValueError(f'{name} is not present in the params dictionary')
        else:
            return par
    
