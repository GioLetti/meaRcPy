#%%
from data_prep.utils import retrieve_cm_stim_data_to_plot,boxplot_stim,scatterplot_stim,plot_cm_analysis,box_plot_sim_analysis,mix_cm_stim_graph,final_results_table,adjust_font_dimension


adjust_font_dimension(title_size=20,legend_size=30,label_size=20,x_ticks=13,y_ticks=13)
#retrieve_cm_stim_data_to_plot(rc_model_analyzed_input_folder='/home/penelope/Desktop/ilya/mearc_model/all_exp_results',spicodyn_results_csv_path='',output_folder='/home/penelope/Desktop/ilya/fin_res_28_12/exp_res_auc',sims=False)

#%%
data_path = '/home/giorgio/Desktop/fin_11_01/sim_res_xrmse/'
output_path = '/home/giorgio/Desktop/fin_11_01/final_sim_exp_graphs_using_xrmse_2/sim_bigger'


# scatterplot_stim(data_path+'stimulation_sum_up.csv',output_path,avg=True,log=True)

box_plot_sim_analysis(data_path+'cm_data_for_boxplot.csv',output_path)

#plot_cm_analysis(data_path+'cm_data_for_scatter.csv',output_path)

# mix_cm_stim_graph(data_path+'stimulation_sum_up.csv',data_path+'cm_sum_up.csv',output_path)
# boxplot_stim(data_path+'stimulation_sum_up.csv',output_path)
# #%%



#final_results_table(data_path+'stimulation_sum_up.csv',data_path+'cm_sum_up.csv',output_path,sim=False)




# RC_AUC vs. CC_AUC: Mann-Whitney-Wilcoxon test two-sided, P_val:1.132e-03 U_stat=1.820e+03
# RC_Pearson vs. CC_Pearson: Mann-Whitney-Wilcoxon test two-sided, P_val:3.920e-04 U_stat=1.864e+03
# RC_AUC vs. TE_AUC: Mann-Whitney-Wilcoxon test two-sided, P_val:1.100e-05 U_stat=1.993e+03
# RC_Pearson vs. TE_Pearson: Mann-Whitney-Wilcoxon test two-sided, P_val:6.336e-13 U_stat=2.417e+03


# %%
