#%%
import nest
import numpy as np
import logging
import sys
import os
import helper
import time




#%%
class simulation:
    def __init__(self,network_dict,simulation_dict,devices_dict,synapse_dicto,output_dir,mea_layout_path):
        
        '''
        Input: 
            - network_dict --> dictionary containing all the structural and spatial infos necessary to build the network
            - simulation_dict --> dictionary containing all the parameters for the kernel and simulation
            - devices_dict --> dictionary containing the infos about all devices used in the network
            - synaptic_dict --> dictionary containing properties of the synapses
            - output_dir --> output directory
            
        '''
        
        #dictionary parsing
        self.network = network_dict
        self.simulation = simulation_dict
        self.devices = devices_dict
        self.synapses = synapse_dicto  
        
        #kernel specification
        self.sim_time = self.simulation['simulation_time'] # simulation time in ms
        self.k_res = self.simulation['kernel_res'] # kernel resolution
        self.seed = self.simulation['seed'] # seed
        self.thr_num = self.simulation['threads_num'] # number of threads used for computation

        # Sampling frequency
        self.samp_freq = helper.time_to_frequency(self.k_res)
        
        #network status dictionary 
        self.network_status={} # it contains all the information regarding the simulation and it will be saved as yaml file at the end
        self.starting_connection = None

        #seed check
        if self.seed==None:
            self.seed = np.random.randint(1,1000000000)
        np.random.seed(self.seed)
        
        
        #output directory creation
        if os.path.exists(output_dir):
            print(f"output folder {output_dir} already exist")
            sys.exit()
        else:
            os.mkdir(output_dir)
        self.out_dir = output_dir
        
        # data folder creation
        self.data_folder_path = os.path.join(self.out_dir,'data')
        os.makedirs(self.data_folder_path)
        # SpiCoDyn data folder
        self.spicodyn_folder = os.path.join(self.data_folder_path,'spicodyn_data_format')
        os.makedirs(self.spicodyn_folder)

        #meta-data folder
        self.meta_data_folder_path = os.path.join(self.out_dir,'metadata')
        os.makedirs(self.meta_data_folder_path)
        helper.save_metadata(self.sim_time,self.samp_freq,self.meta_data_folder_path)

        
        # save a copy of all the parameters (YAML file) used in parameters_copy folder
        params_copy_folder = os.path.join(self.out_dir,'parameters_copy')
        os.makedirs(params_copy_folder)
        helper.save_dict_to_yaml(self.network,params_copy_folder)
        helper.save_dict_to_yaml(self.devices,params_copy_folder)
        helper.save_dict_to_yaml(self.simulation,params_copy_folder)
        helper.save_dict_to_yaml(self.synapses,params_copy_folder)
        
        
        #MEA layout path
        self.mea_layout_path = mea_layout_path
        
        # logging 
        logging_path = os.path.join(self.out_dir,'log_file.log')
        logging.basicConfig(filename=logging_path,level=logging.INFO,format='%(asctime)s %(levelname)s:%(message)s')
        
        logging.info(f"output folder created at {output_dir}")
        

       
    def setup_NEST(self):
        '''
        Set up NEST kernel by resetting the kernel, defining its resolution, seed and number of threads 
        '''
        
        nest.ResetKernel()
        nest.set(resolution=self.k_res,rng_seed = self.seed,local_num_threads=self.thr_num)
        
        logging.info("Set up of NEST kernel complete")
        self.start = time.time()
    
    def start_simulation(self):
        
        '''
        The function starts the simulation in NEST
        '''
        nest.Simulate(self.sim_time)
        self.end = time.time()
        logging.info(f'Simulation completed in {self.end-self.start}')
        
    def arrange_output(self):
        '''
        The function arranges the output obtaine from NEST simulator in the desired type.
        '''

        helper.arrange_output_file(self.out_dir,self.data_folder_path,self.spicodyn_folder,self.samp_freq,self.sim_time)
        
        
    def analysis(self,starting_conn):
        
        helper.avarage_firing_rate(self.out_dir,self.sim_time)
        logging.info(f'AFR anlysis executed')
        helper.retrieve_conn_matrix(self.data_folder_path,self.network['neuron_number'],self.network['population_number'],self.mea_layout_path)
        helper.retrieve_conn_matrix(self.data_folder_path,self.network['neuron_number'],self.network['population_number'],self.mea_layout_path,starting_conn)
    

class MEA_model(simulation):
    
    def __init__(self,network_dict,simulation_dict,devices_dict,synapse_dicto,output_dir,mea_layout_path):
        super().__init__(network_dict,simulation_dict,devices_dict,synapse_dicto,output_dir,mea_layout_path)
        
        '''
        The MEA model class contains the functions necessary to build the network model starting from the dictionaries
        passed to the simulation class instance
        '''
        
        #internal population dicto initialization
        self.populations = {} # dict containing population 
        
        #internal devices dicto 
        self.created_devices  = {}
        
       
        
    def create_pops(self):
        
        '''
        The function extracts the necessary informations from the network dictionary to create the desired network.  
        '''
        
        #parameters parsing
        self.neuron_model = self.network['neuron_model'] # specifications of the neuron model to use to build the population model
        self.neuron_number = self.network['neuron_number'] # neuron number composing each population
        self.populations_number = self.network['population_number'] # population number
        self.stimulated_pops = self.network['stimulated_population'] # number of populations or the exact populations list that are going to be stimulated
        self.isolated_pops = self.network['isolated_population'] # number of isolated clusters of populations
        self.full_inh_pops = self.network['full_inhibitory'] # number of full inhibitory population or the exact population list (all outgoing connections from those populations have negative weights)
        
        
        #! creating populations
        helper.generating_pop(self.neuron_number,self.neuron_model,self.populations_number,self.populations)
        logging.info(f'{self.populations_number} populations generated')


        #! if user has defined isolated_pops
        if self.isolated_pops != 0:
            #define isolated cluster 
            self.clusters =  helper.cluster_gen(self.populations,self.populations_number,self.isolated_pops)
            logging.info(f'Clusters of isolated populations have been created as follow {self.clusters}')
        else:
            self.clusters = None
        
        #! if user has defined full_inhibitory pops
        if self.full_inh_pops != 0:
            # define full inhibitory populations
            self.full_inh = helper.full_inhibitoty_gen(self.populations,self.populations_number,self.full_inh_pops)
            logging.info(f'Full inhibitory populations have been defined as follow {self.full_inh}')
        else:
            self.full_inh = None
            
        #! if user has defined stimulated_pops
        if self.stimulated_pops != 0:
            self.stim_pops = helper.stimulated_pop_gen(self.populations,self.populations_number,self.stimulated_pops)
            logging.info(f'Stimulated populations have been defined as follow {self.stim_pops}')
        else:
            self.stim_pops=None
            
            
    def create_devices(self):
        
        '''
        Creation of the different devices specified in devices dictionary
        '''
        
        available_device = ['noise_generator','poisson_generator','spike_recorder','step_current_generator'] # list of available device
        
        for device in self.devices: # for each device specified in the devices dictionary
            
            if device not in available_device: # if devices is not among the available one
                logging.error(f'{device} is not available. Only {available_device} are usable (check create_devices function).')
                sys.exit(f'{device} is not available.')
                
            else:
                
                if device == 'noise_generator':
                    
                    #noise = nest.Create(device,self.populations_number,self.devices[device],params={'start':100000})
                    noise = nest.Create(device,self.populations_number,params={'std':0.5,'mean':0.,'start':0})
                    self.created_devices['noise_generator']=noise
                    
                if device == 'poisson_generator':
                    
                    #poisson = nest.Create(device,self.populations_number,self.devices[device])
                    poisson = nest.Create('sinusoidal_poisson_generator',self.populations_number,params= 
                                           [{'rate': 6.5,'phase':45,'amplitude':0.45,'frequency':1,'start':0,'stop':300000},
                                            {'rate': 6.5,'phase':45,'amplitude':0.45,'frequency':1,'start':0,'stop':300000},
                                            {'rate': 6.5,'phase':45,'amplitude':0.45,'frequency':1,'start':0,'stop':300000},
                                            {'rate': 6.5,'phase':45,'amplitude':0.45,'frequency':1,'start':0,'stop':300000},
                                            {'rate': 6.5,'phase':45,'amplitude':0.45,'frequency':1,'start':0,'stop':300000},
                                            {'rate': 6.5,'phase':45,'amplitude':0.45,'frequency':1,'start':0,'stop':300000},
                                            {'rate': 6.5,'phase':45,'amplitude':0.45,'frequency':1,'start':0,'stop':300000},
                                            {'rate': 6.5,'phase':45,'amplitude':0.45,'frequency':1,'start':0,'stop':300000},
                                            
                                            ])

                                                                                
                                                                                
                    self.created_devices['poisson_generator']=poisson
            
                if device == 'spike_recorder':
                    
                    spike_rec = nest.Create(device,self.populations_number)
                    self.created_devices['spike_recorder']=spike_rec

                    helper.spike_recorder_handling(spike_rec,self.out_dir)
                    
                if device == 'step_current_generator' and self.stimulated_pops!=0:
                    
                    stim_devices = helper.stimulation_params(self.devices[device],len(self.stim_pops),self.sim_time)
                    self.created_devices['step_current_generator']=stim_devices
                    
        logging.info('all user-defined devices have been created')
        
    
    def create_synapses(self):
        
        '''
        An instance for each synapse specified in the synapses dictionary is created and the name is changed in one of the available synapse to facilitate
        later retrieval and analysis.
        
        Available synapse:
            - noise_syn --> the synapse name of the synapse instance that is going to be used to connect the noise generator to the populations
            - poisson_syn --> the the synapse name of the synapse instance that is going to be used to connect the poisson generator to the populations
            - recorder_syn --> the synapse name of the synapse instance that is going to be used to connect the spike recorder to the populations
            
        '''
        
        available_synapse = ['intra_population_syn','inter_population_syn','noise_syn','poisson_syn','spike_recorder_syn','stimulation_syn'] # list of available synapse
        
        for synapse in self.synapses: # for each synapse specified by the user
            
            if synapse not in available_synapse: # if not in the available synapse, an error is called
                
                logging.error(f'{synapse} is not available. Only {available_synapse} are usable (check create_synapses function)')
                sys.exit(f'{synapse} is not available.')
        
            else:
                
                helper.synapse_copy(self.synapses,synapse) # the instance is created and renamed 
                logging.info(f'synapse instance of type: {synapse} has been created')
        
        logging.info(f'all user defined synapses have been created')

                    
    def connect(self):
        
        '''
        All the connections, both populations and devices, are generated.
        In particular:
            1. Connections among neurons inside a population are generated.
            2. All the different devices are connected to the populations (each device instance is connected to one population).
            3. Connections among different populations are generated. 
        '''
        
        # intra-population connections
        
        intra_syn_type = list(self.synapses['intra_population_syn'].keys())[0]
        
        if intra_syn_type == 'static_synapse':
                
            syn_parameters = self.synapses['intra_population_syn'][intra_syn_type]
            
            # parameters of the uniform distribution from which weights and delays for intra-population synapses are drawn
            low_exc_weight_intra = syn_parameters['low_exc_weight_intra']
            high_exc_weight_intra = syn_parameters['high_exc_weight_intra']
            exc_proportion_intra = syn_parameters['exc_proportion_intra']
            low_inh_weight_intra = syn_parameters['low_inh_weight_intra']
            high_inh_weight_intra = syn_parameters['high_inh_weight_intra']
            low_delay_intra = syn_parameters['low_delay_intra']
            high_delay_intra = syn_parameters['high_delay_intra']

        if intra_syn_type=='stdp_synapse':
            syn_parameters = self.synapses['intra_population_syn'][intra_syn_type]

            tau = syn_parameters['tau_plus']
            lambda_par = syn_parameters['lambda']
            mu_plus = syn_parameters['mu_plus']
            mu_minus = syn_parameters['mu_minus']
            wmax= syn_parameters['Wmax']
        
        stim_counter=0 # counter used to retrieve the current step_current_generator if stimulated populations are present
        
        for num,pop in enumerate(list(self.populations.keys())):
            
            if intra_syn_type == 'static_synapse':
                intra_syn = {'model':'intra_population_syn',
                            'weight':helper.weight_generator(self.neuron_number,self.neuron_number,low_exc_weight_intra,high_exc_weight_intra,exc_proportion_intra,low_inh_weight_intra,high_inh_weight_intra),
                            'delay':np.random.uniform(low_delay_intra,high_delay_intra,size=(self.neuron_number,self.neuron_number)).round(1)}
                #! intra-population connection 
                nest.Connect(self.populations[pop],self.populations[pop],{'rule':'all_to_all','allow_autapses':False},syn_spec=intra_syn) # intra-population connection
           
            elif intra_syn_type=='stdp_synapse':

                perc = 0.75
                exc_syn_num = int(self.neuron_number * (self.neuron_number - 1) * perc)

                inh_syn_num = int(self.neuron_number * (self.neuron_number-1) - exc_syn_num)

                intra_syn_pos = {'synapse_model':'intra_population_syn',
                             'tau_plus':tau,
                             'lambda':lambda_par,
                             'mu_plus':mu_plus,
                             'mu_minus':mu_minus,
                             'Wmax':wmax,
                             'weight':np.reshape(np.random.uniform(9.5,11.5,exc_syn_num),(self.neuron_number,int(exc_syn_num/self.neuron_number)))}

                #! intra-population connection 
                #nest.Connect(self.populations[pop],self.populations[pop],{'rule':'all_to_all','allow_autapses':False},syn_spec=intra_syn) # intra-population connection
                nest.Connect(self.populations[pop],self.populations[pop],{'rule':'fixed_outdegree','outdegree':int(exc_syn_num/self.neuron_number),'allow_multapses':False,'allow_autapses':False},syn_spec=intra_syn_pos) # intra-population connection
  
                intra_syn_neg = {'synapse_model':'intra_population_syn',
                             'tau_plus':tau,
                             'lambda':lambda_par,
                             'mu_plus':mu_plus,
                             'mu_minus':mu_minus,
                             'Wmax': - wmax,
                             'weight':np.reshape(np.random.uniform(-12.5,-0.1,inh_syn_num),(self.neuron_number,int(inh_syn_num/self.neuron_number)))}
                nest.Connect(self.populations[pop],self.populations[pop],{'rule':'fixed_outdegree','outdegree':int(inh_syn_num/self.neuron_number),'allow_multapses':False,'allow_autapses':False},syn_spec=intra_syn_neg) # intra-population connection
  

           
            #! connection to devices
            nest.Connect(self.created_devices['noise_generator'][num],self.populations[pop],syn_spec={'synapse_model':'noise_syn'}) # connect  noise generator to pop
            
            nest.Connect(self.created_devices['poisson_generator'][num],self.populations[pop][0],syn_spec={'synapse_model':'poisson_syn'}) # connect poisson generator to pop
            
            nest.Connect(self.populations[pop],self.created_devices['spike_recorder'][num],syn_spec={'synapse_model':'spike_recorder_syn'}) # connect pop to spike recorder
            
            
            #! connection to stimulator if needed
           
            
            if self.stimulated_pops != 0:
                
                if pop in self.stim_pops:
                    print(f'CHECK POP STIM {pop} {stim_counter}')
                    nest.Connect(self.created_devices['step_current_generator'][stim_counter],self.populations[pop],syn_spec={'synapse_model':'stimulation_syn'})
                    helper.save_stimulation_protocol(self.data_folder_path,self.created_devices['step_current_generator'][stim_counter],pop,stim_counter)
                    stim_counter+=1
                    
        logging.info('Intra-population connections established')
        logging.info('Noise generator connections established')
        logging.info('Poisson generator connections established')
        logging.info('Spike recorder connections established')
        logging.info('Stimulator connections established')
                    
        # inter-population connections
         
        inter_syn_type = list(self.synapses['inter_population_syn'].keys())[0]
        
        if inter_syn_type == 'static_synapse':
            
            
            syn_parameters_inter = self.synapses['inter_population_syn'][inter_syn_type]
            
            # parameters of the uniform distribution from which weights and delays for inter-population synapses are drawn
            low_exc_weight_inter = syn_parameters_inter['low_exc_weight_inter']
            high_exc_weight_inter = syn_parameters_inter['high_exc_weight_inter']
            exc_proportion_inter = syn_parameters_inter['exc_proportion_inter']
            low_inh_weight_inter = syn_parameters_inter['low_inh_weight_inter']
            high_inh_weight_inter = syn_parameters_inter['high_inh_weight_inter']
            low_delay_inter = syn_parameters_inter['low_delay_inter']
            high_delay_inter = syn_parameters_inter['high_delay_inter']
            inter_conn_rule = syn_parameters_inter['rule']
            inter_conn_numb = syn_parameters_inter['inter_conn_number']
            
            helper.connection_matrix(self.populations,inter_conn_rule,self.neuron_number,inter_conn_numb,low_exc_weight_inter,high_exc_weight_inter,exc_proportion_inter,low_inh_weight_inter,high_inh_weight_inter,low_delay_inter,high_delay_inter,self.clusters,self.full_inh)
        logging.info('Inter-population connections established')
            
        
        if inter_syn_type=='stdp_synapse':
            syn_parameters_inter = self.synapses['inter_population_syn'][intra_syn_type]

            tau = syn_parameters_inter['tau_plus']
            lambda_par = syn_parameters_inter['lambda']
            mu_plus = syn_parameters_inter['mu_plus']
            mu_minus = syn_parameters_inter['mu_minus']
            wmax= syn_parameters_inter['Wmax']
            inter_conn_rule = syn_parameters_inter['rule']
            inter_conn_numb = syn_parameters_inter['inter_conn_number']
            
            helper.connection_matrix_plastic(self.populations,inter_conn_rule,self.neuron_number,inter_conn_numb,tau,lambda_par,mu_plus,mu_minus,wmax,self.clusters,self.full_inh)

        starting_connection = nest.GetConnections(synapse_model='inter_population_syn')

        return starting_connection