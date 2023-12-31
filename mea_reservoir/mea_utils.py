#%%
from scipy.stats import halfnorm
import numpy as np
from numpy.linalg import norm
from typing import Union

#%%
def initialize_w_in(m : int, # it is the number of reservoir units 
                  n_ch : int, # it is the number of channel taken from the input data
                  rng : np.random._generator.Generator # rng is a random number generator
                  ) -> np.ndarray:
    
    '''
    Inputs:
        - m = relative size of the micro-units (int value)
        - n_ch = dimension of the macro-network (Basically the number of recorded channel)
        - rng = random number generator
    Output
        - w_in = input weight matrix [n_res x n_ch] 
    '''
    
    n_res = m * n_ch # dimension of the reservoir 

    w_in = np.zeros((n_res,n_ch)) # initialize an empty matrix [n_res x n_ch]

    m_ind = 0 # row index
    
    for ch in range(n_ch): # for each channel we fill the empty matrix

        tmp_ch = halfnorm.rvs(loc=0,scale=1,size=m,random_state=rng) # temporary random array 

        w_in[m_ind:m_ind+m,ch]=tmp_ch/norm(tmp_ch) # we normalize for the L2 norm - This is done as an energy constraint

        m_ind+=m

    return w_in


#%%
def initialize_w(m: int, # number of reservoir units
                 n_ch: int, # number of channel 
                 rng : np.random._generator.Generator # rng is a random number generator
                ) -> np.ndarray :
    '''
    Inputs:
        - m = relative size of the micro-units (int value)
        - n_ch = dimension of the macro-network (Basically the number of recorded channel)
        - rng = random number generator
    Output
        - w = reservoir weight matrix [n_res x n_ch] 
    '''    

    n_res = m *n_ch # dimension of the reservoir

    w = np.zeros((n_res,n_res)) # initialize an empty matrix [n_res x n_res]

    m_ind = 0 # row and column index

    for ch in range(n_ch):

        H = rng.standard_normal((m,m),dtype=np.float32)
        #H = np.random.randn(m, m)
        u, _, vh = np.linalg.svd(H, full_matrices=False)
        mat = u @ vh

        w[m_ind:m_ind+m,m_ind:m_ind+m] = mat

        m_ind+= m

    return w

#%%
def initialize_synaptic_matrix(m: int,
                               n_ch: int,
                               rng : np.random._generator.Generator # rng is a random number generator
                               ) -> np.ndarray:
    '''
    Inputs:
        - m = relative size of the micro-units (int value)
        - n_ch = dimension of the macro-network (Basically the number of recorded channel)
        - rng = random number generator
    Output
        - s = synaptic weight matrix [n_res x n_res]
    '''

    n_res = m * n_ch

    s = np.zeros((n_res,n_res))
    tmp_vector = rng.uniform(0.1,2.01,size=n_res) # values are drawn from uniform distribution

    np.fill_diagonal(s,tmp_vector)

    return s


