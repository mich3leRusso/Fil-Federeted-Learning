import numpy as np 
from typing import Tuple, List
from continuum.datasets import CIFAR100
import os 
def open_dataset(path:str,dataset_name: str ):
    """
    Open the dataset 
    :param path: path of the dataset,
    :param dataset_name: name of dataset (Not implemented ), 
    :return: train and test datasets  
    """

    train_dataset= CIFAR100(path, download=True, train=True).get_data()   
    test_dataset=CIFAR100(path, download=True, train=False).get_data()
    
    return train_dataset, test_dataset 

def split_train_balanced(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_clients: int = 2,
    seed: int = 1, 
    save: str ="" , 
    dataset_name: str= ""
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split dataset into n_clients datasets with balanced number of samples per class.

    :param x_train: Training features, 
    :param y_train: Training labels, 
    :param n_clients: Number of clients, 
    :param seed: Random seed, 
    :param save: path in which save the datasets, 
    :param dataset_name: name of the dataset,
    :return: List of (x_client, y_client)
    """

    classes = np.unique(y_train)
    rng = np.random.RandomState(seed)

    class_indexes = {}
    for c in classes:
        idx = np.where(y_train == c)[0]
        rng.shuffle(idx)
        class_indexes[c] = idx

    client_datasets = []

    for client_id in range(n_clients):
        train_idx = []

        for c in classes:
            idx = class_indexes[c]
            n_samples = len(idx) // n_clients

            start = client_id * n_samples
            end = (client_id + 1) * n_samples if client_id < n_clients - 1 else len(idx)

            train_idx.extend(idx[start:end])

        train_idx = np.array(train_idx)

        x_client = x_train[train_idx]
        y_client = y_train[train_idx]

        
        client_datasets.append((x_client, y_client))
        
        if save:
            os.makedirs(f"{save}/client_{client_id}",exist_ok=True)
            filename_x=f"{save}/client_{client_id}/x_{dataset_name}.npy"
            filename_y=f"{save}/client_{client_id}/y_{dataset_name}.npy"
            
            np.save(filename_x, x_client)
            np.save(filename_y, y_client)
            
    return client_datasets

def run_split(dataset_name: str, save_path: str, path: str="", number_clients: int =1 ):
    """
    Opens the dataset and splits it 
    
    :param dataset_name: name of the dataset you wanna use,
    :param path: path in which you can find the dataset,
    :param number_clients: number of clients, 
    :param save_path: path in which save the splits

    """
    if number_clients <1:
        raise ValueError("number_clients must be >= 1")
    
    os.makedirs(f"{save_path}/server",exist_ok=True)

    train_dataset, test_dataset=open_dataset(path, dataset_name)
    

    split_train_balanced(train_dataset[0], train_dataset[1], number_clients, save=save_path, dataset_name=dataset_name)

    x_test, y_test=test_dataset[0], test_dataset[1]
    
    filename_x=f"{save_path}/server/x_{dataset_name}.npy"
    filename_y=f"{save_path}/server/y_{dataset_name}.npy"
                
    np.save(filename_x, x_test)
    np.save(filename_y, y_test)

    return 
if __name__ == "__main__":
    #usage 
    run_split("CIFAR100", "data_split", "data", 2)
