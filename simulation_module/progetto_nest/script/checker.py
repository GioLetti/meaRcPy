#%%
import os
import matplotlib.pyplot as plt
import seaborn as srs
import pandas as pd
import numpy as np

#path_single='/home/giorgio/Downloads/60_pop_stim_isol/data'
path_single='/home/giorgio/Desktop/nest/progetto_nest/simulation_output/8_pop_stim_isol_full_inh_11/data'
elec_file_list = [f for f in os.scandir(path_single) if f.is_file()]
#%%
lista_dati = []

for ef in elec_file_list:

    elec = pd.read_csv(ef.path,sep='\t')
    
    st = elec['sample_num']
    st = st*0.01
    st = st[(st<300000)&(st>240000)]

    
    #st = np.random.choice(st,int(len(st)))
    lista_dati.append(st)
    


plt.eventplot(lista_dati,linewidths=0.5)

# %%
