




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