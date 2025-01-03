# Simulation with NEST

The simulation module contains scripts and parameters files necessary to run the NEST simulation.\
The spiking Neural Network has been designed to create a computational model of a neuronal culture
reproducing the spiking and bursting behavior shown by in-vitro cultures measured with MEA systems.\
In particular, being interested in network dynamics, a network of point-process neurons tries to replicate the dynamic shown by neuronal culture studied with the MEA system. 

## Install 
To create the environment to run NEST simulation follow the steps below. Start by creating and activating a specific conda environment.

``` conda create --name nest_env python=3.8.20 && conda activate nest_env ```

Install NEST with conda

``` conda install -c conda-forge nest-simulator=3.5 ```
``` conda install ipython, ipykernel ```

## Usage
Follow the steps below to set up the parameters yaml file needed for running the simulations.
The parameters file folder is contained in the simulation module of the GitHub repository.

## Set up parameters files
The four YAML files in the parameters file folder specify different parameters necessary for running the simulations.\
Once loaded in Python the YAML files are formatted as a Python dictionary (or nested dictionary).
Files already present serve as examples and the parameters can be directly modified.\
For all NEST parameters refer to NEST documentation (https://nest-simulator.readthedocs.io/en/v3.3/ref_material/pynest_apis.html)

### network_dict 
The basic network unit is defined as **Population**. A Population is composed of a
variable number of point-process neurons, which emulates a neuronal population close
to an MEA electrode. The **network** is then composed by linking together different **Populations**
The network_dict.yaml file is constituted by six entries:
* neuron_model
  + name: name of the NEST neuron model to be used (izhikevich model has been used and tested, other models are not guaranteed to work)
  + params: parameters of the selected model
* neuron_number: number of neuron composing the **Population**
* population_number: number of **Population**
* stimulated_population: number of **Population**
* isolated_population: number of isolated **Population**, which will form a separate cluster not connected to other **Populations**
* full_inhibitory: number of **Population** whose inter-population connections are all inhibitory

### simulation_dict
Parameters for the NEST kernel:
* simulation_time: simulation time in ms
* kernel_res: resolution of NEST kernel
* seed: seed to use
* threads_num: number of threads to use in NEST computation
* poisson_to_one: if set to True, the poisson generator will be connected to just one neuron in each population. This is based on the idea of single-neuron driving the dynamic. Check references*

### device_dict
The dictionary contains all parameters describing the NEST device used for stimulating and recording the activity of neurons in the simulation. 
* noise_generator: noise generator parameters dictionary
  + mean:  mean expressed in pA
  + std: standard deviation expressed in pA
* poisson_generator: poisson generator parameters dictionary
  + rate: spikes rate expressed as spikes/sec
* spike_recorder: None # spike recorder does not have any parameter
* step_current_generator:
  + background_time: expressed in ms. It represents the length of the simulation in which only background activity will be 
    present
  + stim_time: expressed in ms. It represents the total duration of each stimulation phase
  + stim_num: number of stimulation in each stimulation phase
  + stim_lenght: expressed in ms. It represents the duration of actual stimulation (repeated for stim_num times) during each 
    stimulation phase 
  + stim_amp: expressed in pA. 

### synapse_dict
The dictionary contains synapse parameters for the different types of synapses used in the simulation. In particular, for connecting devices (noise_generator, poisson_generator, step_current_generator, and spike_recorder) only the static_synapse model has been used. For intra-population and inter-population connections, static_synapse and ptsd_synapse models have been used.

* noise_syn: noise generator synapse parameter
  + static_synapse: synapse model name
    + weight: synapse's weight
    + delay: synapse's delay
* poisson_syn: poisson generator synapse parameter
  + static_synapse: synapse model name
    + weight: synapse's weight
    + delay: synapse's delay
* stimulation_syn:  step current generator synapse parameter
  + static_synapse: synapse model name
    + weight: synapse's weight
    + delay: synapse's delay
* spike_recorder_syn: # spike recorder synapse parameter (these values must remain fixed)
  + static_synapse: synapse model name
    + weight: 1.0
    + delay: 1.0
* intra_population_syn:  intra-population synapse parameter
  + static_synapse: synapse model name
    + low_exc_weight_intra: float
    + high_exc_weight_intra: float
    + exc_proportion_intra: float
    + low_inh_weight_intra: float
    + high_inh_weight_intra: float
    + low_delay_intra: float
    + high_delay_intra: float
* inter_population_syn:  inter-population synapse parameter
  + static_synapse: synapse model name
    + low_exc_weight_inter: float
    + high_exc_weight_inter: float
    + exc_proportion_inter: float
    + low_inh_weight_inter: float
    + high_inh_weight_inter: float
    + low_delay_inter: float
    + high_delay_inter: float
    + rule: fixed_outdegree # NEST connection rule
    + inter_conn_number: number of inter-population connection

  ** Intra-population and inter-population weights and delays are drawn from uniform distributions. low and high parameters represent       those uniform distributions' lower and upper bounds (check numpy.random.uniform function)

## MEA layout
In the MEA layout folder, pickle files containing descriptions of the MEA electrodes' position and number can be added based on the desired experimental set-up.
Currently tested and working just for MEA system MEA2100-Mini-Systems of multichannel systems.

## Run simulations
Once the parameters files have been set, go into the script folder of the simulation_module folder. 
The simulation can be run from the terminal as follow:
```
python main.py output_dir_path mea_layout_name (example mea_60.pickle)
```


## References

* 1) Riquelme, Juan Luis, et al. "Single spikes drive sequential propagation and routing of activity in a cortical network." Elife 12 (2023): e79928.
* 2) Hemberger, Mike, et al. "Reliable sequential activation of neural assemblies by single pyramidal cells in a three-layered cortex." Neuron 104.2 (2019): 353-369.





