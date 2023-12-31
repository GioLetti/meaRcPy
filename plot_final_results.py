from data_prep.utils import retrieve_cm_stim_data_to_plot,boxplot_stim,scatterplot_stim,plot_cm_analysis,box_plot_sim_analysis,mix_cm_stim_graph



retrieve_cm_stim_data_to_plot(rc_model_analyzed_input_folder='/home/penelope/Desktop/ilya/mearc_model/all_exp_results',spicodyn_results_csv_path='',output_folder='/home/penelope/Desktop/ilya/fin_res_28_12/exp_res_auc',sims=False)


#data_path = '/home/penelope/Desktop/ilya/mearc_model/final_results_folder/sim_results/cm_data_for_boxplot.csv'
#output_path = '/home/penelope/Desktop/ilya/mearc_model/final_results_folder/sim_results/'


#scatterplot_stim('/home/penelope/Desktop/ilya/mearc_model/final_results_folder/sim_results/stimulation_sum_up.csv','/home/penelope/Desktop/ilya/mearc_model/final_results_folder/tmp_sim',sim=True)
#%%
#box_plot_sim_analysis('/home/penelope/Desktop/ilya/mearc_model/final_results_folder/sim_results/cm_data_for_boxplot.csv','/home/penelope/Desktop/ilya/mearc_model/final_results_folder/sim_results')
#plot_cm_analysis('/home/penelope/Desktop/ilya/mearc_model/final_results_folder/sim_results/cm_data_for_scatter.csv','/home/penelope/Desktop/ilya/mearc_model/final_results_folder/sim_results')
#%%
#mix_cm_stim_graph('/home/penelope/Desktop/ilya/mearc_model/final_results_folder/sim_results/stimulation_sum_up.csv','/home/penelope/Desktop/ilya/mearc_model/final_results_folder/sim_results/cm_sum_up.csv',output_path)
#boxplot_stim('/home/penelope/Desktop/ilya/mearc_model/final_results_folder/sim_results/stimulation_sum_up.csv','/home/penelope/Desktop/ilya/mearc_model/final_results_folder/sim_results')
#%%