from parser import retrieve_parameters,retrieve_params_debug
import mea_snn
import copy

#os.chdir('/home/giorgio/Desktop/progetto_nest/script')

def main():

    # retrieve data
    #output_path,mea_layout_path,network_dict,device_dict,simulation_dict,synapse_dict = retrieve_params_debug()
    output_path,mea_layout_path,network_dict,device_dict,simulation_dict,synapse_dict = retrieve_parameters()

    # Create MEA model 
    mea_model = mea_snn.MEA_model(network_dict,simulation_dict,device_dict,synapse_dict,output_path,mea_layout_path)

    # set up NEST
    mea_model.setup_NEST()

    # generate neuron population, devices, and synapses
    mea_model.create_pops()
    mea_model.create_devices()
    mea_model.create_synapses()

    # connect everything
    starting_connection = mea_model.connect()
    starting_connection = copy.deepcopy([(con.get('source'),con.get('target'),con.get('weight'))for con in starting_connection])
    # start simulation
    mea_model.start_simulation()

    # arrange output
    mea_model.arrange_output()

    # perform AFR analysis
    mea_model.analysis(starting_connection)
        
if __name__ == '__main__':
    main()