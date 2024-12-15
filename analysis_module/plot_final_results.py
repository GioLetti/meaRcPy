#%%
from data_prep.utils import retrieve_cm_stim_data_to_plot,boxplot_stim,scatterplot_stim,plot_cm_analysis,box_plot_sim_analysis,mix_cm_stim_graph,final_results_table,adjust_font_dimension,plot_inhibitory_analysis
import argparse

#%%
################### For reviewers
# To replicate the graphs in the paper it's enough to run two times this script from terminal, one with the 'sim' parameter set to True (for simulation data - default) 
# and the second time with 'sim' set to False (for experimental data)
# I've set the data_path and output_path manually, the only active parameter to be passed when running is the 'sim' parameter.
# ( the parameters 'data_path' and 'output_path' must be passed, but they can be whatever string)
# 
# To abilitate 'data_path' parameter and 'output_path' parameter is enough to comment line from 33 to 40, save the file and run the script from the terminal
##################

if __name__== '__main__':

    parser = argparse.ArgumentParser(description = 'This script is used to gather and plot the analysis analysis data from either experiments or simulations')

    parser.add_argument('data_path',type=str, help='Path to the folder containing the results of the analysis (either simulations or experiments)')
    parser.add_argument('output_path',type=str,help='Path where plots, tables and various csv files will be saved')
    parser.add_argument('sim',type=bool,help='set to True for simulations data, False for experimental data')


    args = parser.parse_args()

    data_path = args.data_path
    output_path = args.output_path
    type_flag = args.sim


    adjust_font_dimension(title_size=18,legend_size=15,label_size=15,x_ticks=13,y_ticks=13)
    #retrieve_cm_stim_data_to_plot(rc_model_analyzed_input_folder='/home/penelope/Desktop/ilya/mearc_model/all_exp_results',spicodyn_results_csv_path='',output_folder='/home/penelope/Desktop/ilya/fin_res_28_12/exp_res_auc',sims=False)

    if type_flag==True: #### comment those lines to abilitate data_path and output_path parameters when running from terminal

        data_path = './data_for_plot/sim/'
        output_path = './data_for_plot/results/sim/'

    else:
        data_path = './data_for_plot/exp/'
        output_path = './data_for_plot/results/exp/'
        ################### till here #################################


    scatterplot_stim(data_path+'stimulation_sum_up.csv',output_path,avg=True,log=True)

    box_plot_sim_analysis(data_path+'cm_data_for_boxplot.csv',output_path)

    plot_cm_analysis(data_path+'cm_data_for_scatter.csv',output_path)

    mix_cm_stim_graph(data_path+'stimulation_sum_up.csv',data_path+'cm_sum_up.csv',output_path)
    boxplot_stim(data_path+'stimulation_sum_up.csv',output_path)
   

    final_results_table(data_path+'stimulation_sum_up.csv',data_path+'cm_sum_up.csv',output_path,sim=type_flag)

    plot_inhibitory_analysis(data_path+'inhibitory_cm_analysis_25_perc.csv',output_path)


#adjust_font_dimension(title_size=25,legend_size=18,label_size=18,x_ticks=15,y_ticks=15)
