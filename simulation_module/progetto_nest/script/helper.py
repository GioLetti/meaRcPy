#%%
import os
import nest
import logging
import numpy as np
import random
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import copy
from itertools import chain
import yaml
#%%

def save_metadata(duration_in_ms,frequency_in_hz,meta_data_path):

    duration_in_sec = duration_in_ms/1000 # simulation duration in sec
    
    df = pd.DataFrame({'recording_duration_sec':[duration_in_sec],'sampling_fr_hz':[frequency_in_hz]})
    df.to_csv(os.path.join(meta_data_path,'meta_data.csv'),sep='\t',index=False)


def spike_recorder_handling(spike_recorder,simulation_path):
    
    '''
    The function defines for each spike recorder the path where to save the recorded spike train
    for each population 
    
    Input
        - spike_recorder --> node collection of the spike recorder 
        - simulation_path --> the general output path of the simulation
    '''
    
    for num,sr in enumerate(spike_recorder):
        
        sr_folder = os.path.join(simulation_path,f'electrode_{num}')
        
        if not os.path.exists(sr_folder):
            
            os.mkdir(sr_folder)
            
            nest.SetStatus(sr,{'record_to':'ascii',
                   'label': os.path.join(sr_folder,f'electrode_{int(num):03d}')})
            

def synapse_copy(synapses,synapse):
    '''
    The function generates a copy of the synapse indicated in the synapses dictionary, but changing the name to a pre-defiened name
    to facilitate later retrievial and analysis
    
    Input
        - synapses --> the synapses dictionary containing user specified parameters for the different synapse
        - synapse --> the new name associated to the copied synapse instance 
    
    '''
    
    synapse_specifications = synapses[synapse]
    
    synapse_type = list(synapse_specifications.keys())[0] # this is the exact NEST name of the synapse 
    synapse_params = synapse_specifications[synapse_type] # this dictionary contains the specified parameters (in the synapse dictionary) of the synapse that we are creating
    
    if synapse != 'intra_population_syn' and synapse != 'inter_population_syn':
     
        nest.CopyModel(synapse_type,synapse,synapse_params) # we create the synapse as specified and rename to facilitate later retrieval
    
    else:
        nest.CopyModel(synapse_type,synapse) 
        
        

    
def weight_generator(n,m,low_uniform_exc,high_uniform_exc,excitatory_proportion,low_uniform_inh,high_uniform_inh):
    
    '''
    Input:
        - n,m --> desired dimension of the weight matrix in output
        - low_uniform_exc,high_uniform_exc --> low and high parameters of uniform distribution from which weight of excitatory synapse are drawn
        - excitatory_proportion --> proportion of excitatory synapses (the % of inhibitory is calculated by subtraction). expected float 0<= x <=1 
        - low_uniform_inh, high_uniform_inh --> low and high parameters for inhibitory synapse
    Output:
        - weights --> nxm matrix of weights
    '''

    
    if type(excitatory_proportion) != float:
        logging.error('float value expected a {} was given'.format(type(excitatory_proportion)))
        raise TypeError('float value expected a {} was given'.format(type(excitatory_proportion)))
        
    if excitatory_proportion > 1 or excitatory_proportion <0 :
        logging.error('float between 0<= x <= 1 is expected')
        raise ValueError('float between 0<= x <= 1 is expected')
    

    entries = n*m # total number of synapses
    
    if entries==1: # if the population is composed by a single neuron
        
        tmp = round(np.random.uniform(),2) # random value between 0 to 0.99
        
        if tmp > (excitatory_proportion-0.1): # if the tmp value is higher than excitatory proportion
            weights = round(np.random.uniform(low_uniform_inh,high_uniform_inh),1) # the synapse weight is negative
        else:
            weights = round(np.random.uniform(low_uniform_exc,high_uniform_exc),1) # the synapse weight is positive
        
        
    else:
        
        num_exc_syns = int(entries * excitatory_proportion) # number of excitatory synapse
        num_inh_syns = entries - num_exc_syns # number of inhibitory synapse
        
        exc_syns = np.random.uniform(low_uniform_exc,high_uniform_exc,size=num_exc_syns) # weights of excitatory synapses
        inh_syns = np.random.uniform(low_uniform_inh,high_uniform_inh,size=num_inh_syns) # weights of inhibitory synapses
        
        mixed = np.reshape(np.concatenate((exc_syns,inh_syns)),(n,m)).round(1) # reshaping to obtain nxm matrix
        
        weights = np.random.permutation(mixed) # a random permutation to mix excitatory and inhibitory synapses
        
    return weights
    

def generating_pop(neuron_number,neuron_model,populations_number,populations):
    
    '''
    The function generates the user-defined nueron populations, which are appended to the populations list.
    
    Input:
        - neuron_number --> user-defined number of neurons in each population
        - neuron_model --> user-defined neuron model variable containing the specification regarding the NEST neuron model name and desired parameters
        - populations_number --> user-defined number of population
        - populations --> the dicto used as container for all created population
    
    Output:
        No output is generated
    
    '''
    
    for pop_id in range(populations_number): # for the number of population
            
            nest.CopyModel(neuron_model['name'],'pop_{}'.format(pop_id)) # a copy of the desired neuron model is created giving a new name to facilitate the later handling
            
            pop = nest.Create('pop_{}'.format(pop_id),neuron_number,params=neuron_model['params']) # the population is created 
            
            populations['pop_{}'.format(pop_id)]=pop # the created population is add to the populations dictionary
            
            #populations_list.append(pop) # the created population is add to the population list
            
            

def cluster_gen(populations,populations_number,isolated_pops):
    
    '''
    The function creates population clusters as defined by the user. Each cluster is composed by at least two populations.
    isolated_pops can be: 
        - an integer (e.g 3) --> three populations clusters will be randomly generated.
        - a list (e.g [3,4,5]) --> three clusters will be created with 3,4,5 populations in each, respectively.
        - a list of list (e.g [["pop_0","pop_1","pop_4"],["pop_2","pop_3"]]) --> two cluster will be generated and composed by the specified populations
    
    Input
        - populations --> the container dicto of populations
        - populations_number --> populations number as user-defined
        - isolated_pops --> user-defined specification of how to create population clusters
    
    Output
        - cluster --> population clusters
    '''
    
    if type(isolated_pops) == int: # if the user as defined only the number of desired cluster, those will be randomly generated
        
        if isolated_pops>(populations_number//2): # population number in each cluster can not be smaller then two.
            logging.error('The selected number of total clusters does not allow to have a minimum of two populations in each cluster. Reduce the cluster value.')
            raise ValueError('The selected number of total clusters does not allow to have a minimum of two populations in each cluster. Reduce the cluster value.')
        else:
            cluster_spec = random_cluster(populations_number,isolated_pops) # this generates n(= isolated_pops) number specifying each cluster composition 
            
            clusters = cluster_builder(populations,cluster_spec) # clusters are generated
            logging.info(f'Populations clusters have been defined as follow {clusters}')
            return clusters
    
    elif type(isolated_pops) == list: # if the user specifies the cluster as list (example: [2,4,5]) or list of list (example: [['pop_0','pop_1'],['pop_3','pop_4','pop_5']])
        
        if type(isolated_pops[0])==int: # if the user has defined the number of populations for each cluster 
        
            if sum(isolated_pops)>populations_number:
                logging.error('The total number of populations considering all cluster is greater than the selected populations number. The sum of the specified list must be equal to "populations_number"')
                raise ValueError('The total number of populations considering all cluster is greater than the selected populations number')
            else:
                clusters = cluster_builder(populations,isolated_pops) # clusters are generated
                logging.info(f'Populations clusters have been defined as follow {clusters}')
                return clusters
        else:
            if sum([len(x) for x in isolated_pops])>populations_number:
                logging.error('The total number of populations considering all cluster is greater than the selected populations number. The sum of the specified list must be equal to "populations_number"')
                raise ValueError('The total number of populations considering all cluster is greater than the selected populations number')
            elif len(min(isolated_pops)) < 2:
                logging.error('The minimum number of populations in a cluster is two. One of the specified cluster does not satisfy this condition.')
                raise ValueError('The minimum number of populations in cluster is two. One of the specified cluster does not satisfy this condition')
            else:
                
                if sum([len(x) for x in isolated_pops])<populations_number: # if there are populations left they will form a unic cluster
                    
                    remaining_pop = [p for p in populations if p not in list(chain(*isolated_pops))]
                    isolated_pops.append(remaining_pop)
                    
                clusters = isolated_pops # in the last case, the user has already specified the clusters as a list of list (desired form)
                return clusters 

    else:
        logging.error('isolated_pops can be an integer (e.g 3), a list (e.g [3,4,5]), or a list of list [["pop_0","pop_1","pop_4"],["pop_2","pop_3"]] (Check cluster_gen function.)')
        raise TypeError('isolated_pops type not correct. Check cluster_gen function.')
    
    
def random_cluster(populations_number,clusters_num):
    '''
    The function generates 'cluster_num' integers specifying each the number of populations for each cluster. 
    
    Input
        - populations_number --> the total number of populations that must be divided in 'cluster_num' clusters.
        - clusters_num --> number of cluster to be generated.
        
    Output
        - cluster_composition --> list of numbers specifying the populations number that will compose each cluster
    
    '''
    
    cluster_composition = [] # list specifying the populations number composing each clusters
    
    for i in range(clusters_num):
        if i == clusters_num - 1:
            value = populations_number - sum(cluster_composition)
        else:
            max_value = populations_number - sum(cluster_composition) - (clusters_num - i - 1)
            if max_value <= 1: # as we can not have cluster of 1 population only
                value = 2
            else:
                value = np.random.randint(2, max_value//2)
        cluster_composition.append(value)
        
    return cluster_composition


def cluster_builder(populations,cluster_spec):
    '''
    Input
        - populations --> the populations dicto
        - cluster_spec --> list containing integer values specifying the number of populations composing each cluster
        
    Output
        - clusters --> list of list where each list contains identifiers of each population inside the cluster
    '''

    clusters=[] # container list for the cluster
    
    pop_list= list(populations.keys()) # list of population identifiers
    
    i=0 # counter
    
    for pop_num in cluster_spec:
        
        tmp = pop_list[i:i+pop_num] # creates a sub-list of population identifiers that is append to the cluster list
        clusters.append(tmp)
        i+=pop_num # increase the counter by the number of population of the previous cluster
        
    return clusters
        
    

def full_inhibitoty_gen(populations,populations_number,full_inh_pops):
    
    '''
    The function creates a list of population identifiers (e.g 'pop_0') for the population that are going to be fully inhibitory (all inter-population connections are negative).
    full_inh_pops can be:
        - an integer (e.g 3) --> in this case three random population will be selected to be fully inhibitory
        - a list (e.g ["pop_0","pop_1","pop_4"]) --> the specified population are going to be fully inhibitory
        - a list composed by [3,["pop_2","pop_3"]] -- > a total of three population will be fully inhibitory and among those two are the ones specified in the list
    
    Input
        - populations --> the container dicto of populations
        - populations_number --> populations number as user-defined
        - full_inh_pops --> user-defined specification of regarding full inhibitory populations
    
    Output
        - full_inh --> list containing identifiers of full inhibitory populations
    '''
    
    
    if type(full_inh_pops)==int: # if the user has defined only the desired number of full inhibitory populations
        
        if full_inh_pops>populations_number:
            logging.error('The desired number of full inhibitory population is greater then the total amount of populations (check full_inhibitory_gen function)')
            raise ValueError('The desired number of full inhibitory population is greater then the total amount of populations')
        
        else:
            full_inh = random.sample(list(populations.keys()),full_inh_pops)
            
            return full_inh
    
    elif type(full_inh_pops)==list: 
        
        if type(full_inh_pops[0]) == int: 
            
            if full_inh_pops[0]>populations_number:
                logging.error('The desired number of full inhibitory population is greater then the total amount of populations (check full_inhibitory_gen function)')
                raise ValueError('The desired number of full inhibitory population is greater then the total amount of populations')
            
            else:
                full_inh = full_inh_pops[1] # a list containing the populations that the user wants to be full inhibitory
                tmp = random.sample([x for x in list(populations.keys()) if x not in full_inh],(full_inh_pops[0]-len(full_inh))) # a list containing the remaining full inhibitory
                full_inh.extend(tmp)
                
                return full_inh
        
        else: # it is already in the desired form. Only some checks are performed
            
            if len(full_inh_pops)>populations_number:
                logging.error('The desired number of full inhibitory population is greater then the total amount of populations (check full_inhibitory_gen function)')
                raise ValueError('The desired number of full inhibitory population is greater then the total amount of populations')
            
            else:
                full_inh= full_inh_pops
                
                return full_inh
    
    else:
        logging.error('full_inh_pops can be an integer (e.g 3), a list (e.g ["pop_0","pop_1","pop_4"]), or a list composed by [3,["pop_2","pop_3"]] (Check cluster_gen function.)')
        raise TypeError('full_inh_pops type not correct. Check full_inhibitory_gen function.')
    

def stimulated_pop_gen(populations,populations_number,stimulated_pops):
    
    '''
    The function creates a list of population identifiers (e.g 'pop_0') for the population that are going to be stimulated.
    if the user has already defined such a list, some checks are performed to ensure correctness.
    
    stimulated_pops can be:
        - an integer (e.g 3) --> in this case three random population will be selected to be stimulated
        - a list (e.g ["pop_0","pop_1","pop_4"]) --> the specified population are going to be stimulated
        - a list composed by [3,["pop_2","pop_3"]] -- > a total of three population will be stimulated and among those two are the ones specified in the list
    
    
    Input
        - populations --> the container dicto of populations
        - populations_number --> populations number as user-defined
        - stimulated_pops --> user-defined specification of regarding stimulated populations
    
    Output
        - stim_pops --> list containing identifiers of populations that are going to be stimulated
    '''
    
    if type(stimulated_pops)==int: # if the user has defined only the desired number of full inhibitory populations
        
        if stimulated_pops>populations_number:
            logging.error('The desired number of stimulated population is greater then the total amount of populations (check stimulated_pop_gen function)')
            raise ValueError('The desired number of stimulated population is greater then the total amount of populations')
        
        else:
            stim_pops = random.sample(list(populations.keys()),stimulated_pops)
            
            return stim_pops
    
    elif type(stimulated_pops)==list: 
        
        if type(stimulated_pops[0]) == int: 
            
            if stimulated_pops[0]>populations_number:
                logging.error('The desired number of stimulated population is greater then the total amount of populations (check stimulated_pop_gen function)')
                raise ValueError('The desired number of stimulated population is greater then the total amount of populations')
            
            else:
                stim_pops = stimulated_pops[1] # a list containing the populations that the user wants to be full inhibitory
                tmp = random.sample([x for x in list(populations.keys()) if x not in stim_pops],stimulated_pops[0]-len(stim_pops)) # a list containing the remaining full inhibitory
                stim_pops.extend(tmp)
                
                return stim_pops
        
        else: # it is already in the desired form. Only some checks are performed
            
            if len(stimulated_pops)>populations_number:
                logging.error('The desired number of stimulated population is greater then the total amount of populations (check stimulated_pop_gen function)')
                raise ValueError('The desired number of stimulated population is greater then the total amount of populations')
            
            else:
                stim_pops= stimulated_pops
                
                return stim_pops
    
    else:
        logging.error('stimulated_pops can be an integer (e.g 3), a list (e.g ["pop_0","pop_1","pop_4"]), or a list composed by [3,["pop_2","pop_3"]] (Check stimulated_pop_gen function.)')
        raise TypeError('stimulated_pops type not correct. Check stimulated_pop_gen function.')


def stimulation_params(stimulus_parameters,stim_pops_num,sim_time):
    '''
    The function defines the time and amplitude array for the step_current_generator that is used as stimulator.
    
    Input
        - stimulus_parameters --> dictionary containing the user-defined parameters for stimulation device
        - stim_pops_num --> the number of stimulated population
        - sim_time --> total simulation time 
       
    Output:
        - stim_device --> Nest node_collection composed by the step_current_generator used as stimulator
    
    '''
    
    background_time = stimulus_parameters['background_time'] # lenght of background activity only phase
    stim_time = stimulus_parameters['stim_time'] # lenght of each stimulation phase
    stim_num = stimulus_parameters['stim_num']
    stim_lenght = stimulus_parameters['stim_lenght'] # expressed in ms
    stim_amp = stimulus_parameters['stim_amp'] # expressed in pA

    # checks    
    if background_time >= sim_time:
        logging.error(f'background time: {background_time} >= simulation time: {sim_time}. Decrease background_time or increase simulation time to perform stimulation analysis')
        raise ValueError(f'background time: {background_time} >= simulation time: {sim_time}.')
        
    if stim_time >= sim_time:
        logging.error(f'Lenght of each stimulation phase: {stim_time} >= simulation time: {sim_time}. Decrease stim_time or increase simulation time to perform stimulation analysis')
        raise ValueError(f'Lenght of each stimulation phase: {stim_time} >= simulation time: {sim_time}.')
        
    if (background_time + stim_time*stim_pops_num) > sim_time:
        logging.error(f'Total lenght of stimulation protocol (background+stimulation phase): {background_time + stim_time*stim_pops_num} >= simulation time: {sim_time}. Decrease stimulation protocol or increase simulation time to perform stimulation analysis')
        raise ValueError(f'Total lenght of stimulation protocol (background+stimulation phase): {background_time + stim_time*stim_pops_num} >= simulation time: {sim_time}.')
    
    
    stim_devices=nest.NodeCollection()
    
    for num_pop in range(stim_pops_num):
        
        on_time = np.arange((background_time+num_pop*stim_time),(background_time+stim_time+num_pop*stim_time),stim_time/stim_num)
        off_time = on_time + stim_lenght
        
        time_array = np.empty((on_time.size + off_time.size,), dtype=on_time.dtype)
        time_array[0::2]=on_time
        time_array[1::2]=off_time
        
        amp_array = [stim_amp,0]*(len(time_array)//2)
        
        
        stim_device=nest.Create('step_current_generator',{'amplitude_times':time_array,'amplitude_values':amp_array})
        
        stim_devices+=stim_device
            
    return stim_devices


def save_stimulation_protocol(data_folder_path,stimulation_device,target_pop,stim_counter):
    '''
    The function saves the stimulation protocol as a dataframe in stimulation_protocol folder created inside the data one
    
    Input
        - data_folder_path --> data folder path
        - stimulation_device --> instance of step_current_generator used as stimulator
        - target_pop --> the target pop identifies that is going to be stimulated 
        - stim_counter --> the stimulation counter
    '''
    
    stimulation_path = os.path.join(data_folder_path,'stimulation_protocols')
    if not os.path.exists(stimulation_path): # it create the stimulation_protocols folder if it does not exist 
        os.mkdir(stimulation_path)
    
    #print(f'CHECK path {stimulation_path}')
    
    amp_times = nest.GetStatus(stimulation_device,'amplitude_times')[0]
    
    on_time = amp_times[::2].tolist() # start of the stimulation 
    off_time = amp_times[1::2].tolist() # end of the stimulation
    
    target_pop = re.findall('\d+',target_pop)[0]
    stimulation_protocol = pd.DataFrame({'start':on_time,'end':off_time,'target':[target_pop]*len(on_time)}) 
    
    stimulation_protocol.to_csv(os.path.join(stimulation_path,f'{stim_counter}_stimulation_protocol.csv'),sep='\t',index=False)
    
        

def connection_matrix(populations,inter_conn_rule,neuron_num,inter_conn,low_exc_weight_inter,high_exc_weight_inter,exc_proportion_inter,low_inh_weight_inter,high_inh_weight_inter,low_delay_inter,high_delay_inter,clusters,full_inh):
    '''
    The function creates the inter-population connection between the different populations.
    
    Input
        - populations --> the container dicto of populations
    
    '''
    
    #! if user has defined cluster of populations 
    if type(clusters) == list:
        
        for cluster in clusters: # for each cluster
            
            for pop in cluster: # for each population in a cluster
                
                targets = pick_pop_to_connect(pop,cluster)
                
                for target in targets: # for each target
                    
                    if type(full_inh) == list and pop in full_inh:
                    
                                    syn_neuron_inter = {'synapse_model':'inter_population_syn',
                                'weight':weight_generator(neuron_num,inter_conn,low_inh_weight_inter,high_inh_weight_inter,exc_proportion_inter,low_inh_weight_inter,high_inh_weight_inter),
                                'delay':np.random.uniform(low_delay_inter,high_delay_inter,size=(neuron_num,inter_conn)).round(1)}
                
                    else:
                        
                        syn_neuron_inter = {'synapse_model':'inter_population_syn',
                                        'weight':weight_generator(neuron_num,inter_conn,low_exc_weight_inter,high_exc_weight_inter,exc_proportion_inter,low_inh_weight_inter,high_inh_weight_inter),
                                        'delay':np.random.uniform(low_delay_inter,high_delay_inter,size=(neuron_num,inter_conn)).round(1)}
                    
                        
                    nest.Connect(populations[pop],populations[target],{'rule':inter_conn_rule,'outdegree':inter_conn,'allow_autapses':False,'allow_multapses':False},syn_spec = syn_neuron_inter)
    
    
    #! if user has not defined cluster of populations                       
    else:
        
        pops = list(populations.keys())
        
        for pop in pops:
            
            targets = pick_pop_to_connect(pop,pops)
            
            for target in targets: # for each target
                    
                if type(full_inh) == list and pop in full_inh:
                
                                syn_neuron_inter = {'synapse_model':'inter_population_syn',
                            'weight':weight_generator(neuron_num,inter_conn,low_inh_weight_inter,high_inh_weight_inter,exc_proportion_inter,low_inh_weight_inter,high_inh_weight_inter),
                            'delay':np.random.uniform(low_delay_inter,high_delay_inter,size=(neuron_num,inter_conn)).round(1)}
            
                else:
                    
                    syn_neuron_inter = {'synapse_model':'inter_population_syn',
                                    'weight':weight_generator(neuron_num,inter_conn,low_exc_weight_inter,high_exc_weight_inter,exc_proportion_inter,low_inh_weight_inter,high_inh_weight_inter),
                                    'delay':np.random.uniform(low_delay_inter,high_delay_inter,size=(neuron_num,inter_conn)).round(1)}
                
                nest.Connect(populations[pop],populations[target],{'rule':inter_conn_rule,'outdegree':inter_conn,'allow_autapses':False,'allow_multapses':False},syn_spec = syn_neuron_inter)



def connection_matrix_plastic(populations,inter_conn_rule,neuron_num,inter_conn,tau,lambda_par,mu_plus,mu_minus,wmax,clusters,full_inh):
    '''
    The function creates the inter-population connection between the different populations.
    
    Input
        - populations --> the container dicto of populations
    
    '''
    
    #! if user has defined cluster of populations 
    if type(clusters) == list:
        
        for cluster in clusters: # for each cluster
            
            for pop in cluster: # for each population in a cluster
                
                targets = pick_pop_to_connect(pop,cluster)
                
                for target in targets: # for each target
                    
                    if type(full_inh) == list and pop in full_inh:
                            
                            

                            syn_neuron_inter_neg = {'synapse_model':'inter_population_syn',
                             'tau_plus':20,
                             'lambda':0.5,
                             'mu_plus':0.1,
                             'mu_minus':0.1,
                             'Wmax':-25,
                             'weight':np.random.uniform(-15,-5,3)}
                            nest.Connect(populations[pop],populations[target],{'rule':'fixed_total_number','N':3,'allow_autapses':False},syn_spec=syn_neuron_inter_neg)
                              

                    else:
                        
                        syn_neuron_inter_pos = {'synapse_model':'inter_population_syn',
                             'tau_plus':20,
                             'lambda':0.5,
                             'mu_plus':0.1,
                             'mu_minus':0.1,
                             'Wmax':20,
                             'weight':np.random.uniform(9.5,10.5,12)}
                       
                        syn_neuron_inter_neg = {'synapse_model':'inter_population_syn',
                             'tau_plus':20,
                             'lambda':0.5,
                             'mu_plus':0.1,
                             'mu_minus':0.1,
                             'Wmax':-25,
                             'weight':np.random.uniform(-15,-5,3)}
                            
                        
                    #nest.Connect(populations[pop],populations[target],{'rule':inter_conn_rule,'outdegree':inter_conn,'allow_autapses':False,'allow_multapses':False},syn_spec = syn_neuron_inter)
                        nest.Connect(populations[pop],populations[target],{'rule':'fixed_total_number','N':12,'allow_autapses':False},syn_spec=syn_neuron_inter_pos)
                        nest.Connect(populations[pop],populations[target],{'rule':'fixed_total_number','N':3,'allow_autapses':False},syn_spec=syn_neuron_inter_neg)
    #! if user has not defined cluster of populations                       
    else:
        
        pops = list(populations.keys())
        
        for pop in pops:
            
            targets = pick_pop_to_connect(pop,pops)
            
            for target in targets: # for each target
                    
                if type(full_inh) == list and pop in full_inh:
                
                            

                            syn_neuron_inter_neg = {'synapse_model':'inter_population_syn',
                             'tau_plus':20,
                             'lambda':0.5,
                             'mu_plus':0.1,
                             'mu_minus':0.1,
                             'Wmax':-25,
                             'weight':np.random.uniform(-15.,-5.,15)}
                            nest.Connect(populations[pop],populations[target],{'rule':'fixed_total_number','N':15,'allow_autapses':False},syn_spec=syn_neuron_inter_neg)
                            
                else:
                    
                    syn_neuron_inter_pos = {'synapse_model':'inter_population_syn',
                             'tau_plus':20,
                             'lambda':0.5,
                             'mu_plus':0.1,
                             'mu_minus':0.1,
                             'Wmax':20,
                             'weight':np.random.uniform(9.5,12.5,12)}

                    syn_neuron_inter_neg = {'synapse_model':'inter_population_syn',
                             'tau_plus':20,
                             'lambda':0.5,
                             'mu_plus':0.1,
                             'mu_minus':0.1,
                             'Wmax':-25,
                             'weight':np.random.uniform(-15.,-5,3)}
                    
                    nest.Connect(populations[pop],populations[target],{'rule':'fixed_total_number','N':12,'allow_autapses':False},syn_spec=syn_neuron_inter_pos)
                    nest.Connect(populations[pop],populations[target],{'rule':'fixed_total_number','N':3,'allow_autapses':False},syn_spec=syn_neuron_inter_neg)
'''
def plastic_synapse(number,tau,lambda_par,mu_plus,mu_minus,wmax,type='intra',sign = 'n'):

    if sign =='n':
        weights = round(np.random.uniform(-2.5,-0.1,size=number),2)
        
        weights = np.reshape(np.concatenate((exc_syns,inh_syns)),(n,m)).round(1)

    else:
        weights = round(np.random.uniform(-2.5,-0.1,size=number),2)
'''


def pick_pop_to_connect(pop,populations):
    
    '''
    The functions generates a list indicating the target populations
    
    Input
        - pop --> the source population
        - populations --> the list of all populations
    
    Output
        - list of target populations
    '''
    tmp = [x for x in populations if x!=pop]  # creates a temporary list without the source population
    
    num_to_connect = np.random.randint(1,(len(tmp)//2+1)) # defines the number of populations that are going to be connected    
    
    #! In case you do (1,(len(tmp)//2+1)) so to not have pop with zero connection you need to handle the tmp=1 case
    
    return random.sample(tmp,k=num_to_connect) # draw the target populations

def time_to_frequency(kernel_resolution):
    '''
    The function convert the kernel resolution (expressed in ms) to sampling frequency (Hz)
    '''
    
    return round(1/(kernel_resolution/1000))
    



def reach_electrode_path(output_dir,exp_elec):
    '''
    The function returns the complete path of the electrode and all the files present in it
    '''
    
    exp_electrode_path = os.path.join(output_dir,exp_elec) #path to each electrode folder
            
    electrode_file_list = os.listdir(exp_electrode_path) # all files recorded by spike recorder for the specific electrode/population taken into considerations    

    return exp_electrode_path,electrode_file_list

def electrode_final_path_builder(exp_electrode,extension=''):
    '''
    The function extracts the electrode number from the name from the output file. If an extension is provided it gives back the name of the output file with the desired extension, 
    otherwise only the number is returned
    '''
    reg = re.compile('\d+') # to idnetify the electrode/population number
    match_reg = re.findall(reg,exp_electrode)[0] 
    
    if extension != '':
        return f'electrode{int(match_reg):03d}.{extension}'
    else:
        return int(match_reg)

def arrange_output_file(output_dir,data_folder_path,spicodyn_path,sampling_frequency,simulation_time):
    
    '''
    The function changes the output style and format
    
    
    Input:
        - experiment_directory --> path of the experiment directory
    '''
    
    
    exp_electrode_list = sorted([x for x in os.listdir(output_dir) if (os.path.isdir(os.path.join(output_dir,x)) and x.startswith('electrode'))]) # list of all electrode's folders in each experiments
    
    for exp_electrode in exp_electrode_list: # for each recorded population/electrode
    
        exp_electrode_path,electrode_file_list  = reach_electrode_path(output_dir,exp_electrode)
        
        dataframe_to_concatenate=[] #a list containing all dataframes to concantenate 
    
        for electrode_file in electrode_file_list:
            
            electrode_file_path = os.path.join(exp_electrode_path,electrode_file) #path to recorded electrode thread
            tmp = pd.read_csv(electrode_file_path,header=2,sep='\t') #read the spike recorder output file
            dataframe_to_concatenate.append(tmp)
            
            #remove original file
            os.remove(electrode_file_path)
        
        
        out_index_file_name = electrode_final_path_builder(exp_electrode,'csv')
        
        output_path = os.path.join(exp_electrode_path,out_index_file_name)
        
        if len(dataframe_to_concatenate)>1: # when using more CPU cores a file for each neuron is created based on the cores number
            df_conc = pd.concat(dataframe_to_concatenate) # if more dataframe are present they are concatenate
        else:
            df_conc = dataframe_to_concatenate[0] # if only one is present (= only one core is used)
    
        df_conc=df_conc.sort_values('time_ms') # the value are sorted based on spiking time (which NEST gives in ms)
        
        df_conc.to_csv(output_path,sep='\t',index=False) # save the data in the specific electrode/population folder
        
        
        df_conc['time_ms']=round(df_conc['time_ms']*(sampling_frequency/1000)) # the spiking time in ms is converted in sample number by multiplying for the sampling frequency
        
        df_conc = df_conc.astype(int)

        df_conc.rename(columns={'time_ms':'sample_num'},inplace=True)
        
        save_spicodyn_format(df_conc['sample_num'],spicodyn_path,electrode_final_path_builder(exp_electrode),sampling_frequency,simulation_time) # save the data in the spicodyn data folder in the format specific for spicodyn
    
        df_conc.to_csv(os.path.join(data_folder_path,out_index_file_name),sep='\t',index=False) # save the data in the data folder



def save_spicodyn_format(series_to_save,spicodyn_path,electrode_number,sampling_frequency,simulation_time):
    '''
    The function save the data in spycodin folder in the format desired by spycodin software for connectivity analysis
    
    Input
        - df_to_save --> dataframe to save
        - spicodyn_path --> path to spicodyn folder 
        - electrode_number --> electrode/population number
        - sampling_frequency --> sampling frequency
        - simulation_time --> simulation time
        
    Output
        - spicodyn file format  
    '''
    
    total_sample_number = (sampling_frequency/1000)*simulation_time # calculates the total number of samples 
    
    total_spike_number = len(series_to_save)
    
    spicodyn_series = pd.concat([pd.Series([total_sample_number,total_spike_number]),series_to_save])
    
    spicodyn_series=spicodyn_series.astype(int)
    
    path_to_save = os.path.join(spicodyn_path,f'electrode_{int(electrode_number):03d}.txt')
    
    spicodyn_series.to_csv(path_to_save,header=None,index=None,sep=' ')
    

def afr_stats(afr_dict,output_dir):
    '''
    The function saves the AFR (spikes/s) of all electrodes/populations
    
    Input
        - afr_dict
    '''
    
    afr_df  = pd.DataFrame(list(afr_dict.items()),columns=['electrode','AFR (spikes/s)'])    

    afr_df.to_csv(os.path.join(output_dir,'afr_stats.csv'),sep='\t',index=False)
    
    

def avarage_firing_rate(output_dir,simulation_time):
    
    
    exp_electrode_list = sorted([x for x in os.listdir(output_dir) if (os.path.isdir(os.path.join(output_dir,x)) and x.startswith('electrode'))]) # list of all electrode's folders in each experiments
    
    simulation_time_s = simulation_time/1000
    
    afr_dict={} # the dictionary contains the AFR expressed in spikes/s for all electrodes/populations
    
    for exp_elec in exp_electrode_list: #for each electrode
        
        electrode_folder_path,electrode_file_list  = reach_electrode_path(output_dir,exp_elec)
        
        file_path = os.path.join(electrode_folder_path,electrode_file_list[0])
        tmp = pd.read_csv(file_path,header=0,sep='\t')
        
        elec_number = electrode_final_path_builder(exp_elec)
               
        number_of_spike = len(tmp)
        
        
        elec_key = f'electrode_{int(elec_number):03d}'
        
        isi = np.diff(np.array(tmp['time_ms'])) # calculates the ISI 
        
        if ((number_of_spike/simulation_time_s)>1 and len(isi)>0):
        
            if elec_key not in afr_dict:
                afr_dict[elec_key]=number_of_spike/simulation_time_s
                
            ISI_plot(isi,electrode_folder_path)
        
        else:
            if elec_key not in afr_dict:
                afr_dict[elec_key]= '< 1 '
    
    afr_stats(afr_dict,output_dir)

   


def ISI_plot(spike_time_stamp,experiment_directory,smoothing_parameter=(7,1)):
    
    '''
    The function produces the log ISI plot and saves it in the experiment directory.
    
    Input:
        - spike_time_stamp --> ISI spike times
        - experiment_directory --> the experiment directory in which plot is saved
        - smoothing_parameter --> smoothing parameter of the Savgol filter applied
    '''
    
    ISI=spike_time_stamp
    ISI = ISI[ISI!=0]
    log_ISI=np.log10(ISI)
    
    
    #Histogram computation log ISI
    histo_bins=np.arange(min(log_ISI),max(log_ISI),0.1)
    histogram=plt.hist(log_ISI,bins=histo_bins,density=True)
    plt.close()
    
    binned_values=np.array(histogram[0]) #amplitude of each log_ISI duration time

    #histogram smoothing applying savgol filter
    window_size,poly_order=smoothing_parameter
    smooth_hist=np.append(np.array(signal.savgol_filter(binned_values,window_size,poly_order)),0)+0.000001#to avoid zero values that can cause problem in void value computation
    
    smooth_hist= np.where(smooth_hist<0,0,smooth_hist)
    
    '''
    #fitting
    
    xdata = histo_bins.round(2)
    xdata=xdata.tolist()
    
    ydata = np.append(binned_values.round(6),0)
    
    #fit gaussian

    bounds=((0,0,0,2.,0,0),(1.5,2,3,3.5,3.5,1))
    params_gauss,pcov_gauss=curve_fit(bimodal_gauss,xdata,ydata,bounds=bounds)
    
    
    values = pd.DataFrame({'x': xdata,'y':ydata})
    values.to_csv(os.path.join(experiment_directory,'ISI_values.csv'),index=False,sep='\t')
    
    
    fit_params = pd.DataFrame({'mu_1':[params_gauss[0]],'sigma_1':[params_gauss[1]],'A_1':[params_gauss[2]],
                               'mu_2':[params_gauss[3]],'sigma_2':[params_gauss[4]],'A_2':[params_gauss[5]]},index=[0])
    fit_params.to_csv(os.path.join(experiment_directory,'fit parameters.csv'),index=False,sep='\t')
    '''   
    
    #plot
    plt.hist(log_ISI,bins=histo_bins,density=True,label='ISI distribution')
    plt.plot(histo_bins,smooth_hist,label='savgol filter smooth')
    #plt.plot(xdata,bimodal_sal(xdata,*params_sal))
    #plt.plot(xdata,bimodal_gauss(xdata,*params_gauss),label='bimodal gaussian fit')
    plt.title('log-ISI distribution')
    plt.xlabel('ISI - log scale')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(experiment_directory,'log_isi.pdf'),format='pdf')
    plt.close()




def retrieve_conn_matrix(data_folder_path,num_neuron,population_num,mea_layout,pre=None):
    
    pickle_in = open(mea_layout,"rb")
    mea_layout = pickle.load(pickle_in)

    xy = [x[1] for x in mea_layout if x[0]<population_num] # xy coordinates of MEA electrodes
    

    #tra_net = os.path.join(tragitto,'network_status.pickle')
    #pickle_in = open(tra_net,"rb")
    #network_conns = pickle.load(pickle_in)
    #inter_conns = network_conns['inter_conn']

    if type(pre)==type(None):
        inter_conns = nest.GetConnections(synapse_model='inter_population_syn') # retrieve all inter-population connection
    else:
        inter_conns = pre
    
    
    connessioni = {} # container dictionary for storing each connction vweight values
    for con in xy:
        connessioni[con]={}


    #calculating firing probability of each neuron in each population
    
    firing_prob = {}
    
    population_firing_data_list = sorted([x for x in os.listdir(data_folder_path) if (os.path.isfile(os.path.join(data_folder_path,x)) and x.startswith('electrode'))])
    

    
    for firing_pop in population_firing_data_list: # for each electrode/population folder in data folder
    
        
        pop_data_path = os.path.join(data_folder_path,firing_pop) # electrode/population path
        
        pop_data = pd.read_csv(pop_data_path,sep='\t') # loading spike data
        
        firing_neuron = pop_data['sender'] # pandas series of which neuron as spiked
        
        
        neurons_firing_prob = firing_neuron.value_counts(normalize=True) # probability of firing for each neuron
        
        neuron_index = neurons_firing_prob.index.to_list() # neuron index (NEST index, so starting from 1)
        neuron_prob = neurons_firing_prob.values.tolist() # neuron probability

        
        
        
        for n_idx in range(len(neuron_index)): 
            
            firing_prob[neuron_index[n_idx]-1] = neuron_prob[n_idx] # neuron_index[n_idx]-1 is used to retrieve the Python index
    
   
    unweighted_connessioni = copy.deepcopy(connessioni) # copy of connesioni 
    
    
    for conn in inter_conns: 
        
        if type(inter_conns) != list:
            sr_id= conn.get('source')-1  #source global id; -1 is used to convert to python index
            tr_id = conn.get('target')- 1 #target global id
                        
            pop_tr = tr_id//num_neuron # trick to identify the target population 
            pop_sr = sr_id//num_neuron # trick to identify the source population
        
            
            if xy[pop_tr] not in connessioni[xy[pop_sr]]:
                connessioni[xy[pop_sr]][xy[pop_tr]]=[conn.get('weight')*firing_prob.get(sr_id,0)] # weight of the connection * firing probability of the source neuron
                unweighted_connessioni[xy[pop_sr]][xy[pop_tr]]=[conn.get('weight')] # weight of the connection
            else:
                connessioni[xy[pop_sr]][xy[pop_tr]].append(conn.get('weight')*firing_prob.get(sr_id,0))
                unweighted_connessioni[xy[pop_sr]][xy[pop_tr]].append(conn.get('weight'))
        else:
            sr_id= conn[0]-1  #source global id; -1 is used to convert to python index
            tr_id = conn[1]- 1 #target global id
                        
            pop_tr = tr_id//num_neuron # trick to identify the target population 
            pop_sr = sr_id//num_neuron # trick to identify the source population
        
            
            if xy[pop_tr] not in connessioni[xy[pop_sr]]:
                connessioni[xy[pop_sr]][xy[pop_tr]]=[conn[2]*firing_prob.get(sr_id,0)] # weight of the connection * firing probability of the source neuron
                unweighted_connessioni[xy[pop_sr]][xy[pop_tr]]=[conn[2]] # weight of the connection
            else:
                connessioni[xy[pop_sr]][xy[pop_tr]].append(conn[2]*firing_prob.get(sr_id,0))
                unweighted_connessioni[xy[pop_sr]][xy[pop_tr]].append(conn[2])





    for x in connessioni:
        for y in connessioni[x]:
            connessioni[x][y]=np.mean(connessioni[x][y]) # average connections weight for the connection*firing probability
            unweighted_connessioni[x][y]=np.mean(unweighted_connessioni[x][y]) # average connections weight 
            
        
    '''        
    all_values = list(connessioni.values())
        
    min_list=min([x[y] for x in all_values for y in x])
    max_list=max([x[y] for x in all_values for y in x])
        
    norm = Normalize(min_list, max_list)
    
    for x in connessioni:
        for y in connessioni[x]:
            connessioni[x][y]=norm(connessioni[x][y])
    '''

    # create connectivity matrix folder in data folder
    conn_matrix_path =os.path.join(data_folder_path,'connectivity_matrices')
    if not os.path.exists(conn_matrix_path):
        os.mkdir(conn_matrix_path)
    
    if type(pre)==type(None):
        file_name = 'weighted_conn_matrix.csv'
        file_name_base = 'base_conn_matrix.csv'
    else:
        file_name = 'initial_weighted_conn_matrix.csv'
        file_name_base = 'initial_base_conn_matrix.csv'

    weighted_conn = pd.DataFrame(connessioni)
    weighted_conn=weighted_conn.reindex(xy)
    weighted_conn.to_csv(os.path.join(conn_matrix_path,file_name),sep='\t')  
    
    base_conn = pd.DataFrame(unweighted_connessioni)
    base_conn=base_conn.reindex(xy)
    base_conn.to_csv(os.path.join(conn_matrix_path,file_name_base),sep='\t')  




def plot_connectivity_matrix(csv_file):
    # Read the connectivity matrix from the CSV file
    connectivity_matrix = pd.read_csv(csv_file, header=None).values

    # Get the number of nodes
    num_nodes = len(connectivity_matrix)

    # Create a scatter plot
    fig, ax = plt.subplots()

    # Iterate over each connection in the connectivity matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            # If there is a connection from node i to node j
            if connectivity_matrix[i, j] == 1:
                # Draw an arrow from node i to node j
                ax.arrow(i, j, 0.5, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')

    # Set axis labels
    ax.set_xlabel('Nodes')
    ax.set_ylabel('Nodes')

    # Set the x and y axis limits
    ax.set_xlim([-0.5, num_nodes - 0.5])
    ax.set_ylim([-0.5, num_nodes - 0.5])

    # Set the aspect ratio to be equal
    ax.set_aspect('equal')

    # Show the plot
    plt.show()
    
    
def save_dict_to_yaml(data, file_path):
    '''
    The function saves a copy of each parameters dictionary back to a YAM file as a copy inside the parameters_copy folder
    '''
    # check that data is a dictionary
    if type(data)!=dict:
        raise TypeError('The data must be a dictionary')
    
    
    if 'simulation_time' in data: # it means that data is simulation_dict
        file_path = os.path.join(file_path,'copy_simulation_dict.yaml')
    elif 'neuron_model' in data: # it means that data is network_dict.yaml
        file_path = os.path.join(file_path,'copy_network_dict.yaml')
    elif 'spike_recorder' in data: # it means that data is device_dict.yaml
        file_path = os.path.join(file_path,'copy_device_dict.yaml')
    elif 'spike_recorder_syn' in data: # it means that data is synapse_dict.yaml
        file_path = os.path.join(file_path,'copy_synapse_dict.yaml') 
    else:
        raise NotImplementedError('The dictionary has not been recognized among the available (simulation_dict,network_dict,device_dict, and synapse_dict). Please check save_dict_to_yaml function')

    with open(file_path, 'w') as file:
        yaml.dump(data, file)
        


#%%



#%%
'''
path = '/home/giorgio/Downloads/16_pop_vanilla_2/data'
path='/home/giorgio/Desktop/nest/progetto_nest/simulation_output/ciao_ciao/data'
elec_file_list = [f for f in os.scandir(path) if f.is_file()]

lista_dati = []

for ef in elec_file_list:

    elec = pd.read_csv(ef.path,sep='\t')
    st = elec['sample_num']
    st = st*0.01
    st = st[st<10000]
    lista_dati.append(st.values)
    
plt.eventplot(lista_dati,linewidths=1)

#%%

path_real = '/home/giorgio/Desktop/stim_data/results_2023_08_08-12_23/30-09-2022/41436_14DIV/41436_14DIV_D-00144'


elec_file_list = [f for f in os.scandir(path_real) if f.is_file()]

lista_dati = []

for ef in elec_file_list:

    elec = pd.read_csv(ef.path,sep='\t')
    st = elec['sample_num']
    st = st*0.05
    st = st[(st<60000)&(st>0)]
    lista_dati.append(st.values)
    

plt.eventplot(lista_dati,linewidths=0.5)

#%%
path_single='/home/giorgio/Downloads/single_neuron_vanilla_1/data'
elec_file_list = [f for f in os.scandir(path_single) if f.is_file()]

lista_dati = []

for ef in elec_file_list:

    elec = pd.read_csv(ef.path,sep='\t')
    st = elec['sample_num']
    st = st*0.01
    st = st[st<60000]
    lista_dati.append(st.values)
    


plt.eventplot(lista_dati,linewidths=0.5)
'''
'''
# %%
import os
import matplotlib.pyplot as plt
import seaborn as srs
import pandas as pd
import numpy as np

#path_single='/home/giorgio/Downloads/60_pop_stim_isol/data'
path_single='/home/giorgio/Desktop/nest/progetto_nest/simulation_output/long_sim_single_poiss_4/data'
elec_file_list = [f for f in os.scandir(path_single) if f.is_file()]

lista_dati = []

for ef in elec_file_list:

    elec = pd.read_csv(ef.path,sep='\t')
    
    st = elec['sample_num']
    st = st*0.01
    st = st[(st<420000)&(st>360000)]

    
    #st = np.random.choice(st,int(len(st)))
    lista_dati.append(st)
    


plt.eventplot(lista_dati,linewidths=0.5)

# %%
import re
path_alg = '/home/giorgio/Downloads/Table_of_CC_TE.txt'
path_rc = '/home/giorgio/Downloads/Table_model_results.txt'

df_alg = pd.read_csv(path_alg,sep=',')
df_rc = pd.read_csv(path_rc,sep=',')
result={}

col_name_alg =list(df_alg.columns)
col_name_rc = list(df_rc.columns)

# alg value
for x in col_name_alg:
    id = re.findall('(?<=x)\d*',x)
    if len(id)>0:
        val_cc = df_alg[x][0]
        val_te = df_alg[x][1]
        if id[0] not in result:
            result[id[0]]={'CC':[val_cc],'TE':[val_te],'RC':[]}
        else:
            result[id[0]]['CC'].append(val_cc)
            result[id[0]]['TE'].append(val_te)
    elif 'prova' in x:
        result['60']['CC'].append(df_alg[x][0])
        result['60']['TE'].append(df_alg[x][1])
# rc value
pop_list=['4','32','16','8','60']
for x in col_name_rc:
    id = re.findall('\d*(?=_)',x)
    if len(id)>0:
        if id[0] in pop_list:
            val_rc = df_rc[x][0]
            result[id[0]]['RC'].append(val_rc)
    elif 'prova' in x:
        result['60']['RC'].append(df_rc[x][0])


f_df_mean={'4':[],'8':[],'16':[],'32':[],'60':[]}
f_df_std={'4':[],'8':[],'16':[],'32':[],'60':[]}

correct_order = ['4','8','16','32','60']
for co in correct_order:
    pop = result[co]

    te_mean = np.mean(pop['TE'])
    cc_mean = np.mean(pop['CC'])
    rc_mean = np.mean(pop['RC'])
    f_df_mean[co].extend((te_mean,cc_mean,rc_mean))
    f_df_std[co].extend((np.std(pop['TE']),np.std(pop['CC']),np.std(pop['RC'])))
#%%

TE_std = [0.512155,	0.155134,	0.140571,	0.086031,	0.097747]
TE_mean = [0.433421,	0.177657,	0.189028,	0.179529,	0.132265]
CC_std=[0.036404,	0.112415,	0.076798,	0.147259,	0.139228]
CC_mean=[0.693624,	0.711584,	0.700273,	0.392764,	0.166603]
rc_std=[0.079031,	0.044220,	0.097113,	0.211174,	0.179808]
rc_mean=[0.850534,	0.817438,	0.784277,	0.732585,	0.409344]
#%%
f_df_mean = pd.DataFrame(f_df_mean,index=['TE','CC','RC'])
f_df_std = pd.DataFrame(f_df_std,index=['TE','CC','RC'])

#%%

#f_df_mean.T.plot(kind='bar', title ="Average Correlation", figsize=(15, 10), legend=True, fontsize=12,xlabel='pop',ylabel='rho',yerr=f_df_std)

#%%

fig, ax = plt.subplots()


ind = np.arange(5)    # the x locations for the groups
width = 0.25         # the width of the bars
ax.bar(ind,TE_mean,width, yerr=TE_std,label='TD')
ax.bar( ind+width,CC_mean, width,yerr=CC_std,
       label='CC')
ax.bar( ind+2*width,rc_mean,width,yerr=rc_std,
       label='RC')

ax.set_title('Average correlation')
ax.set_xticks(ind + width , labels=['4', '8', '16', '32', '60'])

#%%


#plt.savefig('/home/giorgio/Desktop/nest/results/avg_rho_pop.pdf')


f_df_mean.to_csv('/home/giorgio/Desktop/nest/results/avg_rho_pop.csv',',')
f_df_std.to_csv('/home/giorgio/Desktop/nest/results/std_rho_pop.csv',',')
#%%

fin_df = pd.DataFrame(result)


# %%
'''