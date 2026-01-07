import numpy as np

def load_dataset(path:str, dataset_name: str)-> np.ndarray:
    """
    :param path: path in which are stored the data,
    :param dataset: name of the dataset we are using,
    :return: numpy array for train labels and train data 

    """
    x_filename=f"{path}/x_{dataset_name}.npy"
    y_filename=f"{path}/y_{dataset_name}.npy"
    
    x_train=np.load(x_filename)
    y_train=np.load(y_filename)
    
    return x_train, y_train

if __name__=="__main__":
    #usage example
    path="data_split/client_0"
    dataset_name="CIFAR100" 
    
    x_client, y_client = load_dataset(path, dataset_name)

    path="data_split/server"
    x_server, y_server = load_dataset(path, dataset_name)

    print(y_client.shape, y_server.shape)
    