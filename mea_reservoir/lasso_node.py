
'''
author: Giorgio Letti
Licence: MIT License
Copyright: Giorgio Letti (2023) <giorgio.letti@iit.it>

lasso is a subclass builded from the Node class of reservoirpy (check: https://github.com/reservoirpy/reservoirpy/tree/master/reservoirpy)
using the LassoCV function of scikit-learn (check: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)

'''
#%%
#from sklearnex import patch_sklearn
#patch_sklearn()
from functools import partial
from typing import Union
import numpy as np
from reservoirpy.node import Node
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit

from joblib import parallel_backend
from profiler_utility import profile,print_stats
#from ray.train.sklearn import SklearnTrainer
#from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import Lasso
#from sklearnex.linear_model import Lasso
#from sklearnex import config_context


#%%

class lasso(Node):
    '''
    Single layer of neurons learning using lasso regression.
    As Lasso does not have close solution, coordinate gradient is used during learning. 
    The class uses LassoCV function from sklearn (check https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)

    Input:
        - x: matrix of reservoir neuron states of dimension t x n_res, where t = NBs_number * NB_window_lenght.
        - y: matrix of ground truth data of dimension n_ch x t. The ground truth is constitued be the spike counts of binned NB windows
        - rng: either a np random number generator or an int to istantiate one. If nothing is passed 1 will be used as seed to ensure consistency.

    Additional argument: 
        - eps: Lasso path length. 
        - lambdas_par: either a list or np.array of lambda (alpha in scikit nomenclature) parameter to be used in fitting lasso. Default None
        - n_lambdas: total number of lambda parameter to be used. If not specified the default of scikit will be used (100)
        - cv: a cross-validation generator. It must be CV specific for timeseries data. Default is None and in that case a TimeSeriesSplit is going to be internally generated.
        - tol: tolerance for optimization. If not specified the default value of scikit will be used (1e-4)
        - fit_intercept: whether to calculate the intercept for the lasso. Default False as in scikit.
        - max_iter: maximum number of iterations in the fitting of Lasso. Default 1000 as in scikit
        - n_jobs: number of workers to use during cv. Default None. if set to -1 all available will be used.

    '''


    def __init__(self,
                 name:Union[None,str]=None,
                 seed:Union[None,int] = None,
                 **kwargs):
                
                
                # every other argument are arguments for the LassoCV function of scikit-learn
                #possible_argument_list = ['n_jobs','max_iter','fit_intercept','tol','cv','n_lambdas','lambdas_par','eps'] # possible additional arguments

                
                #for key in list(kwargs.keys()):
                #      if key not in possible_argument_list:
                #            print(f'Argument {key} not recognized. Discarded.')
                #            del kwargs[key] # removing not recognized argument

                # cross-validator
                if isinstance(kwargs.get('cv'),type(None)): # if no cv has been passed, a timeseriessplit is created
                      cv = TimeSeriesSplit()
                      kwargs['cv']=cv

                # rng handling
                if seed == None: # if seed is not specified, 1 will be used to ensure consistency
                    seed = 1
                


                lasso_model = _lasso_initializer(seed=seed,**kwargs) # initialize the Lasso
                
                super(lasso, self).__init__(params={'lasso':lasso_model,'w_out':None,'bias':None},
                                hypers={'seed':seed,
                                        'n_jobs':kwargs.get('n_jobs'),
                                        'eps':kwargs.get('eps'),
                                        'tol':kwargs.get('tol'),
                                        'max_iter':kwargs.get('max_iter'),
                                        'fit_intercept':kwargs.get('fit_intercept'),
                                        'cv':kwargs.get('cv'),
                                        'n_lambdas':kwargs.get('n_lambdas'),
                                        'lambdas_par':kwargs.get('lambdas_par')},
                name=name)
                self._is_initialized=True
                
    #@profile
    def fit(self,
            x:np.ndarray,
            y:np.ndarray):
            '''
            The function is used to fit the lasso model
            '''

            if not isinstance(x,list):
                raise TypeError('x must be a np ndarray of dimension NBs_number x NB_window_lenght x n_res. x is composed by reservoir neuron states.')

            if not isinstance(y,list):
                raise TypeError('y must be a np array of dimensions n_ch x (NBs_number * NB_window_lenght). y is composed by the spikes count calculated in each bin of each NB window.')

            lasso_model = self.get_param('lasso')

            x =  np.concatenate(x,axis=0) # x is a np matrix of reservoir neurons states calculated over all NB windows. x has dimension (NBs_number * NB_window_lenght) x n_res
            y = np.concatenate(y,axis=1) # y is a np matrix of the ground truth value, represented by the NB window expressed as spikes number. y as dimension n_ch x (NBs_number * NB_window_lenght)

            w_out = np.zeros((y.shape[0],x.shape[1])) # w_out has dimension n_ch x n_res 
            if self.get_param('fit_intercept'):
                bias = np.zeros(y.shape[0]) # biases
           
            
            for ch in range(y.shape[0]): # for each channel a lasso model is fitted
                
                    fitted_lasso=lasso_model.fit(x[0:-1,:],y[ch,1:]) # y is staggered of one index compared to x 
                
                    w_out[ch,:] = fitted_lasso.coef_

                    if self.get_param('fit_intercept'):
                        bias[ch]=fitted_lasso.intercept_
            
            
            self.set_param('w_out',w_out)
            if self.get_param('fit_intercept'):
                 self.set_param('bias',bias)
            #print_stats() 
        
    def forward(self,
            x:np.ndarray) -> np.ndarray:
      
            w_out = self.get_param('w_out')

            if self.get_param('fit_intercept'):
                bias = self.get_param('bias')
            else:
                bias = 0

            return np.squeeze(w_out @ x.T) + bias


  

def _lasso_initializer(seed:Union[None,int]=None, # random number generator
                    eps:float = 1e-3, # path length (lambda_min / lambda_max
                    lambdas_par: Union[np.ndarray,list,None] = None, # custom array of the lambda parameter of Lasso where to compute the model (in scikit lambda is referred as alpha)
                    n_lambdas: int=100, # number of lambdas used to train the model
                    cv: Union[None,TimeSeriesSplit]=None, # cross-validation strategy splitting. It must be a timeseries cv splitter
                    tol: float = 1e-5, # tolerance for the lasso optimization
                    fit_intercept: bool = True, # whether to calculate the intercept
                    max_iter:int = 1000, # maximum number of iterations
                    n_jobs:Union[None,int]=None, # number of workers to use during cross-validation
                    **kwargs):
    '''
    Wrapper function used to initialize a LassoCV instance. The arguments follow the ones described in the scikit-learn function LassoCV.

    Input:
        - rng: random number generator 
        - eps: 
        - lambdas_par: either a list or np.array of lambda (alpha in scikit nomenclature) parameter to be used in fitting lasso. Default None
        - n_lambdas: total number of lambda parameter to be used. If not specified the default of scikit will be used (100)
        - cv: a cross-validation generator. It must be CV specific for timeseries data. Default is None and in that case a TimeSeriesSplit is going to be internally generated.
        - tol: tolerance for optimization. If not specified the default value of scikit will be used (1e-4)
        - fit_intercept: whether to calculate the intercept for the lasso. Default False.
        - max_iter: maximum number of iterations in the fitting of Lasso. Default 1000 as in scikit
        - n_jobs: number of workers to use during cv. Default None. if set to -1 all available will be used.
        - eps: Lasso path length
    '''


    lasso_model = LassoCV(eps=eps,n_alphas=n_lambdas,alphas=lambdas_par,cv=cv,fit_intercept=fit_intercept,tol=tol,max_iter=max_iter,random_state=seed,n_jobs=n_jobs,precompute=True)

    return lasso_model




    
      
    






'''
def lasso_fit(node:Lasso,
              x:np.ndarray, # training data
              y:np.ndarray, # ground truth
              *args):
    
    The function is used to fit the lasso model
    

    if not isinstance(x,np.ndarray):
          raise TypeError('x must be a np ndarray of dimension NBs_number x NB_window_lenght x n_res. x is composed by reservoir neuron states.')

    if not isinstance(y,np.ndarray):
          raise TypeError('y must be a np array of dimensions n_ch x (NBs_number * NB_window_lenght). y is composed by the spikes count calculated in each bin of each NB window.')

    lasso_model =  node.get_param('lasso')

    x =  np.concatenate(x,axis=0) # x is a np matrix of reservoir neurons states calculated over all NB windows. x has dimension (NBs_number * NB_window_lenght) x n_res
    y = np.concatenate(y,axis=1) # y is a np matrix of the ground truth value, represented by the NB window expressed as spikes number. y as dimension n_ch x (NBs_number * NB_window_lenght)

    w_out = np.zeros((y.shape[0],x.shape[1])) # w_out has dimension n_ch x n_res 

    for ch in range(y.shape[0]): # for each channel a lasso model is fitted
           
        lasso_model.fit(x[0:-1,:],y[ch,1:]) # y is staggered of one index compared to x 
        w_out[ch,:] = lasso_model.coeff_
    
    node.set_param('w_out',w_out)
    
'''    