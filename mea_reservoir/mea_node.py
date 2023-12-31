#%%
'''
@ author: Giorgio Letti

mea_reservoir (mea_res) is a subclass builded from the Node class of reservoirpy (check https://github.com/reservoirpy/reservoirpy/tree/master/reservoirpy)

'''

from typing import Any, Dict, Union, Callable
import numpy as np
from reservoirpy import Node

from reservoirpy.activationsfunc import get_function,tanh
from mea_reservoir.mea_utils import initialize_w_in,initialize_w,initialize_synaptic_matrix
from functools import partial


def half_tanh(x:np.ndarray):

    res = np.tanh(x)
    hv = np.heaviside(res,0)
    return res * hv



class mea_res(Node):

    '''
    The update rule of the reservoir node follows:

    \\mathbf{x}[t] = f (\\mathbf{S} \\cdot (\\mathbf{W}_{in} \\cdot \\mathbf{u}[t] + \\alpha * \\mathbf{W} \\cdot \\mathbf{x}[t-1]))
         
    where:
        - :math:`\\mathbf{x}` is the output activation vector of the reservoir;
        - :math:`\\mathbf{u}` is the input timeseries;
        - :math:`\\mathbf{W}_{In}` is the input weights matrix 
        - :math:`\\mathbf{W}` is the reservoir weights matrix
        - :math:`\\mathbf{S}` is the synaptic weight matrix (it represents a plasticity term )
        - :math:`f` is an activation function (generally tanh)
        - :math: `\\alpha` is a memory term 

    The mea_res.params are the following: (params are mutable - learned during training)
        - W_In: Input weights matrix
        - W: Reservoir weights matrix 
    
    The mea_res.hypers are the following: (hypers are fixed):

        - S: synaptic weight matrix, it is a diagonal matrix with values drawn from a uniform dist ]0,1]
        - alpha: memory term between [0,1], it indicates the importance of the previous state in the calculation of the actual state of the reservoir
        - m : number of reservoir units 
        - n_ch : number of MEA channel
        - seed : the desired seed which will be used to instantiate a numpy random generator istance. If not specified 1 will be used as default.
        - activation : the desired activation function used to update the reservoir nodes state. If not specified, tanh will be used as default.
    '''


    def __init__(self,
                m: int = None, # number of reservoir units
                n_ch: int = None, # number of mea channels
                activation: Union[str, Callable] = half_tanh, # the desired activation function for reservoir layer (Default: tanh)
                rng: Union[None,int,np.random._generator.Generator] = None, # if seed is not specified, 1 will be used to avoid problem with scipy
                alpha: Union[None,float] = None, # if alpha is not specified, it will be randomly set between [0,1]
                syn: Union[np.ndarray,Callable] = None, # synaptic matrix can be passed
                name=None,
                **kwargs):
        
        # check that m = int
        if not isinstance(m,int):
            raise TypeError ('m (reservoir units) must be an integer value')
        
        # check that n_ch = int
        if not isinstance(n_ch,int):
            raise TypeError ('n_ch (MEA channel) must be an integer value')


        # retrieve the desired activation function (if not specified tanh will be used)
        if isinstance(activation,str):
            activation= get_function(activation)

        if rng == None: # if seed is not specified, 1 will be used to ensure consistency
            rng = np.random.default_rng(1)
        elif isinstance(rng,int):
            rng = np.random.default_rng(rng)
            

        # setting alpha value if None
        if alpha == None:
            alpha=np.round(rng.random(1),1)[0]
        
        else:
            alpha=np.round(alpha,1)

        # create synaptic matrix (it's an hyper parameter)
        
        if isinstance(syn,np.ndarray):
            syn_mat = syn
        else:
            syn_mat = initialize_synaptic_matrix(m,n_ch,rng)

        
        super(mea_res,self).__init__(params={'reservoir_weights': None,
                                 'input_weights': None},
                        hypers={'alpha': alpha,
                                'synaptic_matrix': syn_mat,
                                'rng': rng,
                                'm':m,
                                'n_ch':n_ch,
                                'activation':activation},
                        forward=forward,
                        initializer=partial(initialize,**kwargs),
                        output_dim=m*n_ch,
                        name=name)

def forward(node: mea_res, x: np.ndarray) -> np.ndarray:
    

    state = node.state().T  # get current node state
    
    u=x.T # input data 

    alpha = node.get_param('alpha')
    syn_mat = node.get_param('synaptic_matrix')

    w_in = node.get_param('input_weights') # input weights matrix
    w = node.get_param('reservoir_weights') # reservoir weights matrix_generator

    f = node.get_param('activation') # activation function


    return f(syn_mat @ (w_in.dot(u) + alpha * w.dot(state))).T


def initialize(node: mea_res, 
               x: np.ndarray = None,
               w: Union[np.ndarray,Callable] = None,
               w_in: Union[np.ndarray,Callable] = None,
               **kwargs):
     '''
     This function receives a data point x at runtime and uses it to
     infer input and output dimensions.
     Moreover it is used to initialize some parameters of the node istance
     '''

     if x is not None:

        # retrieve necessary hyperparameters
        m = node.get_param('m')
        n_ch = node.get_param('n_ch')
        rng = node.get_param('rng')


        # checks to ensure that data have same MEA channel as defined by n_ch
        if n_ch == x[0].shape[0]:

            node.set_input_dim(x[0].shape[0])
            node.set_output_dim(m*n_ch) # for each channel we have m units that forms the micro-reservoir
        else:
            raise ValueError ('The expected number of channel in x must be equal to n_ch. Check the data structure')

       
        # setting params

        if isinstance(w,np.ndarray):
            node.set_param('reservoir_weights',w)
        elif isinstance(w,type(callable)):
            node.set_param('reservoir_weights',w)
        else:
            node.set_param('reservoir_weights',initialize_w(m,n_ch,rng))

        if isinstance(w_in,np.ndarray):
            node.set_param('input_weights', w_in)
        elif isinstance(w_in,type(callable)):
            node.set_param('input_weights', w_in)
        else:
            node.set_param('input_weights',initialize_w_in(m,n_ch,rng))
        


        
        
    