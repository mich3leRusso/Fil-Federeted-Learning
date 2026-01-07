from argparse import ArgumentParser
from omegaconf import DictConfig
import numpy as np


from utils.transforms import default_transforms

from utils.configuration_utils import load_configuration

def build_client_model() -> DictConfig:
    
    parser = ArgumentParser()
    parser.add_argument("base_path", type=str, help="base folder configuration path")
    parser.add_argument("config_file", type=str, help="specific file configuration")

    args = parser.parse_args()
    base_path = f"{args.base_path}/base.yaml"
    client_path = f"{args.base_path}/clients/client.yaml"

    config_parameters = load_configuration(base_path, args.config_file, client_path)


    if config_parameters["control"] == 0:
        config_parameters.extra_classes = config_parameters["classes_per_exp"] * (config_parameters["class_augmentation"] - 1)
    else:
        config_parameters.extra_classes = 0
    
    print(config_parameters)
    return config_parameters

def build_server_model() ->DictConfig:
    
    parser = ArgumentParser()
    parser.add_argument("base_path", type=str, help="base folder configuration path")

    args = parser.parse_args()
    base_path = f"{args.base_path}/base.yaml"
    server_path=  f"{args.base_path}/server.yaml"
    config_parameters = load_configuration(base_path, args.config_file, server_path)

    return config_parameters

# global declaration
config_parameters = build_client_model()
