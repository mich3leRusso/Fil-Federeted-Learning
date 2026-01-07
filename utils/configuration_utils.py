from omegaconf import OmegaConf, DictConfig
from typing  import Dict
import os 

def load_configuration(config_path:str, node_path: str, client_path:str="") -> DictConfig:
    """
    Load the configuration passed to the client 

    :param config_path: path of the configuration
    :param node_path: configuration specific to the node
    :param client_path: specific configuration for all the clients  
    :return: configuration structure

    """
    base_cfg = OmegaConf.load(config_path)


    node_cfg= OmegaConf.load(node_path)
    
    if client_path:
        client_cfg=OmegaConf.load(client_path)
        conf = OmegaConf.merge(base_cfg, node_cfg, client_cfg)
    
    else:
        conf = OmegaConf.merge(base_cfg, node_cfg)
    

    return conf

def create_config(dir_path: str, filename:str, dictionary: Dict) -> None:
    """
    Create a YAML configuration file and save it to the given path.

    Parameters
    ----------
    path : str
        Output path of the YAML file.
    filename : str
        filename 
    dictionary : Dict
        Configuration data.
    """
    os.makedirs(dir_path, exist_ok=True)

    full_path = os.path.join(dir_path, filename)
    
    cfg = OmegaConf.create(dictionary)
    
    OmegaConf.save(cfg, full_path)

    return 

if __name__=="__main__":
    
    #usage example 
    #create a client configuration
    config_path="configurations"
    filename="client_1.yaml"
    client_config={"client_id":1 }
    create_config(config_path, filename, client_config)

    #load the client details
    path="configurations/base.yaml"
    client_path=f"{config_path}/{filename}"
    cgf= load_configuration(path, client_path)
    print(cgf["scheduler_distillation"][0])

