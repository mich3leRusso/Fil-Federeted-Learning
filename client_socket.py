import json
import asyncio
import websockets
from omegaconf import DictConfig
import os 
import numpy as np
import random
from continuum import ClassIncremental
from continuum.datasets import InMemoryDataset
from utils.transforms import default_transforms
from mind import MIND 
from models import gresnet32, gresnet18, gresnet18mlp
from parser import build_client_model
import torch 
from copy import deepcopy
from utils.generic import freeze_model, set_seed, setup_logger
from torch.utils.data import DataLoader
from continuum.tasks import split_train_val

POLL_INTERVAL = 1.0   # seconds between "ask ready?" polls
TOTAL_TASKS=5


def build_request_position(task_id: int):
    return json.dumps({
        "status": 1,
        "ask": 0,
        "task": task_id,
        "network_position": None
    })


def build_send_network(task_id: int, position: str):
    return json.dumps({
        "status": 2,
        "ask": 0,
        "task": task_id,
        "network_position": position
    })


def build_check_ready(task_id: int):
    return json.dumps({
        "status": 1,
        "ask": 1,
        "task": task_id,
        "network_position": None
    })


def parse_message(raw: str) -> dict:
    return json.loads(raw)


async def train_local_model():
    """Simulate model training."""
    print("Training local model...")
    await asyncio.sleep(2)
    return "local_model_weights"


class FederatedClient:

    def __init__(self, config_parameters: DictConfig):
        #instantiate the model and the model
        #  
        self.config_parameters = config_parameters

        #comunication parameters 
        self.server_uri =config_parameters["SERVER_URI"] 
        self.client_id= config_parameters["client_id"]
        
        #Setting 
        self.dataset= config_parameters["dataset"]
        
        #classes 
        self.n_classes= config_parameters["n_classes"]
        self.n_experiences = config_parameters["n_experiences"]
        self.classes_per_exp=  config_parameters["classes_per_exp"]
    
        #model details
        self.model=config_parameters["model"]
        self.device=config_parameters["device"]
        
        # log files 
        setup_logger()

        #create the network
        self.create_network()
        self.load_dataset()

        #load dataset 
    def create_network(self):
        
        #network initialization
        if self.model == 'gresnet32':
            model = gresnet32(dropout_rate = self.config_parameters["dropout"])
        elif self.model == 'gresnet18':
            model = gresnet18(num_classes=self.n_classes)
        elif self.model == 'gresnet18mlp':
            model = gresnet18mlp(num_classes=self.n_classes)
        else:
            raise ValueError("Model not found.")

        model.to(self.device)
        
        self.strategy= MIND(model)

             
    def load_dataset(self):
        
        #Open the dataset 
        data_path=f"{self.config_parameters['dataset_path']}/client_{self.client_id}"
        dataset=os.listdir(data_path)
        
        #load data 
        x_train=np.load(os.path.join(data_path,dataset[0]))
        y_train=np.load(os.path.join(data_path, dataset[1]))
        
        class_order = list(range(100))
        random.shuffle(class_order)
        

        train_dataset = InMemoryDataset(x_train, y_train)

        self.strategy.train_scenario = ClassIncremental(
            train_dataset,
            increment=self.classes_per_exp,
            class_order=class_order,
            transformations=default_transforms)
        
     


    async def run(self):

        async with websockets.connect(self.server_uri) as ws:
            task_id=0

            while task_id < TOTAL_TASKS:
                

                # ------------------------------------------------------
                # 1. Request aggregator position
                # ------------------------------------------------------
                
                await ws.send(build_request_position(task_id))
                print(" Requested aggregator position")

                resp = parse_message(await ws.recv())
                
                print(" Aggregator position:", resp.get("aggregator_pos"))

                # ------------------------------------------------------
                # 2. Train the network locally
                # ------------------------------------------------------
                
                #======================
                #TAKE THE TASK DATA 
                #======================
                train_taskset=self.strategy.train_scenario[task_id]

                self.strategy.experience_idx = task_id
                self.strategy.model.set_output_mask(task_id, train_taskset.get_classes())

                # prepare dataset

                self.strategy.train_taskset, self.strategy.val_taskset = split_train_val(train_taskset, val_split=self.config_parameters.val_split)
                self.strategy.train_dataloader = DataLoader(self.strategy.train_taskset, batch_size=self.config_parameters.bsize, shuffle=True)

                train_taskset=self.strategy.train_scenario[task_id]
                
                #======================
                #START THE TRAIN OF THE TEACHER  
                #======================
                
                
                        
                if not self.config_parameters.self_distillation:
                    if self.model == 'gresnet32':
                        self.strategy.fresh_model = gresnet32(dropout_rate=self.config_parameters.dropout)
                    elif self.model == 'gresnet18':
                        self.strategy.fresh_model = gresnet18(num_classes=self.config_parameters.n_classes)
                    elif self.model == 'gresnet18mlp':
                        self.strategy.fresh_model = gresnet18mlp(num_classes=self.config_parameters.n_classes)
                    else:
                        raise ValueError("Model not found.")
                else:
                    self.strategy.fresh_model = deepcopy(config_parameters.model)
                    self.strategy.distillation = False
                    self.strategy.pruner.set_gating_masks(self.strategy.fresh_model, self.strategy.experience_idx, weight_sharing=config_parameters.weight_sharing, distillation=self.strategy.distillation)
                
                self.strategy.fresh_model.to(self.config_parameters.device)
                self.strategy.fresh_model.set_output_mask(task_id, train_taskset.get_classes())
                
                # instantiate optimizer 
                self.strategy.train_epochs = self.config_parameters.epochs
                self.strategy.distillation = False
                self.strategy.optimizer = torch.optim.AdamW(self.strategy.fresh_model.parameters(), lr=self.config_parameters.lr, weight_decay=self.config_parameters.wd)
                self.strategy.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.strategy.optimizer, milestones=self.config_parameters.scheduler, gamma=0.5, last_epoch=-1, verbose=False)

                self.strategy.train()
                
                #======================
                #START THE DISTILLATION
                #======================

                self.strategy.distill_model = freeze_model(deepcopy(self.strategy.fresh_model))
                self.strategy.distill_model.to(self.device)
        
                with torch.no_grad():
                    self.strategy.pruner.prune(self.strategy.model, self.strategy.experience_idx, self.strategy.distill_model, self.config_parameters.self_distillation)


                self.strategy.train_epochs = self.config_parameters.epochs_distillation
                self.strategy.distillation = True
                self.strategy.optimizer = torch.optim.AdamW(self.strategy.model.parameters(), lr=self.config_parameters.lr_distillation, weight_decay=self.config_parameters.wd_distillation)
                self.strategy.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.strategy.optimizer, milestones=self.config_parameters.scheduler_distillation, gamma=0.5, last_epoch=-1, verbose=False)
                print(f"    >>> Start Finetuning epochs: {self.config_parameters.epochs_distillation} <<<")
                self.strategy.pruner.set_gating_masks(self.strategy.model, self.strategy.experience_idx, weight_sharing=self.config_parameters.weight_sharing, distillation=self.strategy.distillation)
                
                self.strategy.train()


                local_weights = await train_local_model()
                #save the weights of the network
                 
                # ------------------------------------------------------
                # 3. Send the position of the  network to the server
                # ------------------------------------------------------
                await ws.send(build_send_network(task_id, self.config_parameters.client_models))

                resp = parse_message(await ws.recv())
                print(" Server received model:", resp.get("received"))

                # ------------------------------------------------------
                # 4. Poll server: “Is aggregation done?”
                # ------------------------------------------------------
                print("Waiting for server aggregator...")

                while True:
                    # Ask “ready?”
                    await ws.send(build_check_ready(task_id))
                    raw = await ws.recv()
                    resp = parse_message(raw)

                    if "ready" in resp and resp["ready"] == 1:
                        print(" Aggregation completed!")
                        break

                    elif "error" in resp:
                        print(" Server error:", resp["error"])
                        break

                    await asyncio.sleep(POLL_INTERVAL)

                task_id+=1
                print(" task finished")


if __name__ == "__main__":
    
    config_parameters=build_client_model()
    client = FederatedClient(config_parameters)
    input() 
    asyncio.run(client.run())   
