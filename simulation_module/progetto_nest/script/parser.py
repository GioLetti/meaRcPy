#%%
import argparse
import yaml
import os

#%%

def load_yaml_file(file_path):
    
    # check if the path exist
    if not os.path.exists(file_path):
        raise argparse.ArgumentError(f'Path {file_path} does not exist.')
    
    # Check if the path leads to a file
    if not os.path.isfile(file_path):
        raise argparse.ArgumentError(f'Path {file_path} is not a file')
    
    # Check if the file has a YAML extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in ('.yaml','.yml'):
        raise argparse.ArgumentTypeError(f'File {file_path} is not a YAML file.')
    
    
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


def check_layout(file_path,mea_layout_path):
   
    # check if mea_layout folder exist, if not it is created and print an error
    if not os.path.exists(file_path):
        
        os.makedirs(file_path)
        raise argparse.ArgumentError(f'mea layout folder does not exist! no MEA layout available. Folder has been created, add a MEA layout file (format .pickle)')
    
    if not os.path.exists(mea_layout_path):
        raise argparse.ArgumentError(f'Path: {mea_layout_path} does not exist')
     
    if not os.path.isfile(mea_layout_path):
        raise argparse.ArgumentError(f'Path: {mea_layout_path} is not a file ')
    
    # Check if the file has a csv extension
    _, ext = os.path.splitext(mea_layout_path)
    if ext.lower() not in ('.pickle'):
        raise argparse.ArgumentTypeError(f'File {mea_layout_path} is not a pickle file.')
    

def create_parser():

    parser = argparse.ArgumentParser(description='The program allows for the simulation of an approximation of MEA recording system.')
    
    #arguments
    
    parser.add_argument('output_dir',type=str,help='select the name of the output directory in which simulation files are going to be saved. The folder will be created in the simulation_output folder. ')
    parser.add_argument('MEA_layout',type=str,help='''one of the MEA layout (.pickle format) present in the folder mea_layout. Available:\n\t- mea_60: MEA 2100 mini with 60 electrodes''')
    
    return parser


def retrieve_parameters():
    '''
    The function retrieve all the parameters YAML file converting them to python dictionary and take the output directory name from the argparse
    '''
    # retrieve output directory name
    par = create_parser()
    
    args = par.parse_args()
    
    output_name = args.output_dir
    mea_channel_layout = args.MEA_layout
    
    output_path = '../simulation_output'
    output_abs_path = os.path.abspath(output_path)
    output_abs_path = os.path.join(output_abs_path,output_name)
    
    mea_channel_path = '../mea_layout'
    mea_channel_abs_path = os.path.abspath(mea_channel_path)
    mea_channel_abs_path = os.path.join(mea_channel_abs_path,mea_channel_layout)
    check_layout(mea_channel_path,mea_channel_abs_path)
    
    # retrieve YAML files
    
    parameters_folder_path = '../parameters_file_folder'
    
    parameters_folder_path = os.path.abspath(parameters_folder_path)
    
    network_dict = load_yaml_file(os.path.join(parameters_folder_path,'network_dict.yaml'))
    device_dict = load_yaml_file(os.path.join(parameters_folder_path,'device_dict.yaml'))
    simulation_dict= load_yaml_file(os.path.join(parameters_folder_path,'simulation_dict.yaml'))
    synapse_dicto = load_yaml_file(os.path.join(parameters_folder_path,'synapse_dict.yaml'))
    
    
    return output_abs_path,mea_channel_abs_path,network_dict,device_dict,simulation_dict,synapse_dicto


def retrieve_params_debug():
    
    net = load_yaml_file('/home/giorgio/Desktop/nest/progetto_nest/parameters_file_folder/network_dict.yaml')
    dev = load_yaml_file('/home/giorgio/Desktop/nest/progetto_nest/parameters_file_folder/device_dict.yaml')
    sim =load_yaml_file('/home/giorgio/Desktop/nest/progetto_nest/parameters_file_folder/simulation_dict.yaml')
    syn =load_yaml_file('/home/giorgio/Desktop/nest/progetto_nest/parameters_file_folder/synapse_dict.yaml')
    
    output_path = '/home/giorgio/Desktop/nest/progetto_nest/simulation_output'
    output_abs_path = os.path.abspath(output_path)
    output_abs_path = os.path.join(output_abs_path,'single_poiss_noise_cazzo-finaleee')
    
    mea_channel_path = '/home/giorgio/Desktop/nest/progetto_nest/mea_layout'
    mea_channel_abs_path = os.path.abspath(mea_channel_path)
    mea_channel_abs_path = os.path.join(mea_channel_abs_path,'mea_60.pickle')
    
    return output_abs_path,mea_channel_abs_path,net,dev,sim,syn