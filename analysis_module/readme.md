**Installation** //
To create the environment for running the MeaRcPy model and analysis follow the below steps./

'''
conda create --name mearcpy python=3.8.20 && conda activate mearcpy
'''

After creating and activating the conda environment we can install the packages using pip/

'''
pip install reservoirpy==0.3.9
pip install ipython, ipykernel
pip install seaborn, statsmodels, statannotations
pip install scikit-learn
'''
/

The analysis_example jupyter notebook gives an example of the full analysis pipeline, step by step, over a simulated dataset.
You should look at this jupyter notebook to understand the MeaRcPy analysis workflow.

To reproduce the graphs in the paper is enough to run the script plot_final_results.py.

To run the analysis over multiple data the main.py script can be run.

Have fun!
