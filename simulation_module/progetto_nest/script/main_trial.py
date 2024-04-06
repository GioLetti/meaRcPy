import argparse
import yaml
import os

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



def validate_output_directory(output_path):
    
    # Check if the path already exists
    if os.path.exists(output_path):   
        raise argparse.ArgumentError(f"Output directory '{output_path}' already exist. Choose another name")
    
    else:
        # create the directory
        os.makedirs(output_path)
            
    return output_path



def create_parser():

    parser = argparse.ArgumentParser(description='The program allows for the simulation of an approximation of MEA recording system.')
    
    #arguments
    parser.add_argument('network',type=load_yaml_file,help= 'Path to the network YAML file.')
    parser.add_argument('devices',type=load_yaml_file,help= 'Path to the devices YAML file.')
    parser.add_argument('simulation',type=load_yaml_file,help= 'Path to the simulation YAML file.')
    parser.add_argument('synapses',type=load_yaml_file,help= 'Path to the synapses YAML file.')
    parser.add_argument('output_dir',type=)

    
    return parser



    
args = parser.parse_args()

network_dicto = args.network
devices_dicto = args.devices
simulation_dicto = args.simulation
synapses_dicto = args.synapses