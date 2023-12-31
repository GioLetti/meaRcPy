#%%

from mea_reservoir.mea_node import mea_res
from mea_reservoir.lasso_node import lasso
from reservoirpy.model import FrozenModel
from reservoirpy import node
import numpy as np
from typing import Union
from joblib import Parallel,delayed
from data_prep.metric import squared_error
from profiler_utility import profile,print_stats
from reservoirpy.activationsfunc import relu


#%%
_LEARNING_METHODS = {'lasso': lasso}
_RESERVOIR_METHODS = {'mea_res':mea_res}


# %%
class meaRC(FrozenModel):

    '''
    The meaRC model is a wrapper for mea_node and lasso node.

    optional argument:
        - alpha: float. if specified the value will be used as alpha parameter for mea_res node
        - optimize_alpha: boolean

    '''

    def __init__(self,
                tr:np.ndarray, # training data
                val:np.ndarray, # validation data
                m:Union[None,int]=None, # m parameter of mea reservoir node
                reservoir_method='mea_res',
                learning_method="lasso",
                reservoir: node = None,
                readout: node = None,
                name:str = None,
                rng:Union[None,int,]=1, # an int which will be used as seed and to create a np.random number generator which will be used truoghout the code . If nothing is passed 1 will be used as seed to ensure consistency.
                n_jobs:int=1, # number of workers to use. -1 to use all available workers
                **kwargs):
        
        #_MEA_RES_OPTIONAL = ['alpha']   
        #_LASSO_OPTIONAL = ['fit_intercept']     

        self.results = {'validation':{},'testing':{}} # dictionary in which results will be stored

        #rng handling
        if rng == None: # if seed is not specified, 1 will be used to ensure consistency
            rng,seed = np.random.default_rng(1),1
        elif isinstance(rng,int):
            seed,rng = rng,np.random.default_rng(rng) # both rng and seed will be used as sk-learn does not allow to use a np.default_rng istance

        #data handling
        if not isinstance(tr,list):
            raise ValueError(f'x (training data) must be a np.ndarray')
        else:
            self.tr = tr
        
        if not isinstance(val,list):
            raise ValueError(f'y (ground truth), if passed, must be a np.ndarray')
        else:
            self.val = val

        #optional argument handling
        #for key in list(kwargs.keys()):
        #   if key not in _MEA_RES_OPTIONAL:
        #       del kwargs[key]

 
        msg = "'{}' is not a valid method. Available methods for {} are {}."

       

        if reservoir is None: # if a mea reservoir is not passed at initialization one is created using m=1 
            if reservoir_method not in _RESERVOIR_METHODS:
                raise ValueError(
                    msg.format(reservoir_method, "reservoir", list(_RESERVOIR_METHODS.keys()))
                )
            else:
                if (isinstance(_RESERVOIR_METHODS.get(reservoir_method),type(mea_res)) and (isinstance(m,type(None)))):
                    raise ValueError(f'If {reservoir_method} has not been created before meaRC creation, m must be passed as argument')
                else:
                    reservoir = _RESERVOIR_METHODS.get(reservoir_method)(m=m,n_ch=tr[0].shape[0],rng=rng,**kwargs)
        else:
            if type(reservoir) not in list(_RESERVOIR_METHODS.values()):
                raise msg.format(reservoir_method, "reservoir", list(_RESERVOIR_METHODS.keys()))
                
        if readout is None: # same for lasso
            if learning_method not in _LEARNING_METHODS:
                raise ValueError(
                    msg.format(
                        learning_method, "readout", list(_LEARNING_METHODS.keys())
                    ))
            else:
                readout = _LEARNING_METHODS.get(learning_method)(seed=seed,n_jobs=n_jobs,**kwargs)
        else:
            if type(readout) not in list(_LEARNING_METHODS.values()):
                raise ValueError(msg.format(learning_method, "readout", list(_LEARNING_METHODS.keys())))

      

        
        super(meaRC, self).__init__(
                nodes=[reservoir, readout], edges=[(reservoir, readout)], name=name
            )
        
        self._hypers.update(
            {
                "n_jobs": n_jobs,
                "reservoir_method": reservoir.name,
                "learning_method": readout.name,
                "rng":rng, 
                'seed':seed
            }
        )
    


    def get_hyper(self,name):
        return self.hyper.get(name)
    

    def _calculate_states(self,data):
        '''
        calculate reservoir neuron state 
        '''

        mea_node = self.get_node(self.hypers['reservoir_method'])
        n_jobs = self.hypers['n_jobs']

        if not mea_node.is_initialized: # if reservoir node is not initialized 
            mea_node.initialize(data)

        mea_node.reset()

        with Parallel(n_jobs=n_jobs) as parallel: # parallel computing using joblib
            states = parallel(delayed(mea_node.run)(nb.T,reset=True) for nb in data) # reservoir neuron states are calculated
        #states= np.array(states)

        mea_node.reset()

        return states

    def _connectivity_matrix(self):
        '''
        The function calculates the connectivity matrix after training of the meaRC model. The CM is stored in the results dictionary under the key 'cm'
        '''

        mea_node = self.get_node(self.hypers['reservoir_method'])
        lasso_node = self.get_node(self.hypers['learning_method'])
        w_out = lasso_node.get_param('w_out') # output matrix learned with lasso
        w_in = mea_node.get_param('input_weights') # input weight matrix of mea_node
        syn = mea_node.get_param('synaptic_matrix') # synaptic matrix of mea_node

        cm = w_out @ syn @ w_in # connectivity matrix

        self.results['cm']=cm # connectivity matrix is saved
    
    def _fit(self):
        '''
        Train meaRC model on training data. W-out matrix is saved in the hypers dictionary under the key 'w_out'
        '''
        
        ns_tr = self._calculate_states(self.tr) # reservoir neuron states are calculated passing to the model the training data

        lasso_node = self.get_node(self.hypers['learning_method']) # retrieve lasso node
        lasso_node.fit(ns_tr,self.tr) # performs lasso fit
        w_out = lasso_node.get_param('w_out')
        
        
    
    def _validate(self,
                  metric:str = 'rmse' # root mean squared error
                  ):
        '''
        validation of the meaRC trained model on validation data. MSE  are used as validation metrics, which are stored in the results dictionary under keys 'mse' and 'mape', respectively.
        '''

        ns_val = self._calculate_states(self.val) # reservoir neuron states calculated on validation data with trained meaRC model

        ns_val = np.concatenate(ns_val,axis=0) # concatenate the neuron states to obtain a single matrix


        w_out = self.get_node(self.hypers['learning_method']).get_param('w_out') # output weight matrix

        prediction = w_out @ ns_val.T # predicted spike activity
        prediction +=  1e-16 # to avoid zero division

        val = np.concatenate(self.val,axis=1)
        val += 1e-16

        tmp = np.sum((val + prediction),axis=1)
        weight_component = np.abs(np.divide((prediction + val),tmp[:,np.newaxis]))
        
        err = squared_error(val,prediction) 
        weighted_err = np.sqrt(np.sum(weight_component * err,axis=1)) # weighted RMSE
        weighted_err = np.mean(weighted_err)
        self.results['validation'][metric] = weighted_err
       

        
    
    def run(self):
        '''
        Run meaRC model, including both training and validation.
        '''
        
        self._fit() # training
        self._validate() # validation
        self._connectivity_matrix() # CM calculation
        
    def stimulate(self,
                  stimulation_vector:np.ndarray,
                  steps_number:int=100
                  ):
        '''
        Stimulate the trained meaRC model
        '''

        mea_node = self.get_node(self.hypers['reservoir_method'])
        lasso_node = self.get_node(self.hypers['learning_method'])

        #check stimulation vector dimension corresponds with the model one
        n_ch = mea_node.get_param('n_ch')
        if n_ch != stimulation_vector.shape[0]:
            raise ValueError(f'expected dimension for stimulation vector is {n_ch}. Get {stimulation_vector.shape[0]}.')

        predicted_psth = np.zeros((stimulation_vector.shape[0],steps_number))

        mea_node.reset()
        for step in range(steps_number):
            if step == 0:
                states = mea_node(stimulation_vector) 
            else:
                states = mea_node(np.squeeze(predicted_psth[:,step-1]))

            predicted_step = np.squeeze(lasso_node.forward(states))

            predicted_step = relu(predicted_step)

            predicted_psth[:,step] = predicted_step
        
        mea_node.reset()

        return predicted_psth

        
