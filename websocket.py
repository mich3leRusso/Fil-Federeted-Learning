from typing import List, Dict
from fastapi import WebSocket
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
from parser import build_server_model
from utils.generic import freeze_model, set_seed, setup_logger
from models import gresnet32, gresnet18, gresnet18mlp
from mind import MIND
from copy import deepcopy

class ConnectionManager:
    def __init__(self,config_params) -> None:
        self.config_params= config_params 
        self.active_connections: List[WebSocket] = []
        self.trained_networks: List[dict] = []
        self.tasks: Dict[int, int] = {}   # task_id -> ready_status (0/1)
        
        #create the central model
    async def create_central_model(self):
        
        set_seed(self.config_params.seed)     
        
        # crete the model
        if self.config_params.model == 'gresnet32':
            model = gresnet32(dropout_rate = self.config_params.dropout)
        elif self.config_params.model == 'gresnet18':
            model = gresnet18(num_classes=self.config_params.n_classes)
        elif self.config_params.model == 'gresnet18mlp':
            model = gresnet18mlp(num_classes=self.config_params.n_classes)
        else:
            raise ValueError("Model not found.")
        
        model.to(self.config_params.device)

        self.central_model=MIND(model)
        

        if not self.config_params.self_distillation:
            if self.config_params.model == 'gresnet32':
                self.central_model.fresh_model = gresnet32(dropout_rate = self.config_params.dropout)
            elif self.config_params.model == 'gresnet18':
                self.central_model.fresh_model = gresnet18(num_classes=self.config_params.n_classes)
            elif self.config_params.model == 'gresnet18mlp':
                self.central_model.fresh_model = gresnet18mlp(num_classes=self.config_params.n_classes)
            else:
                    raise ValueError("Model not found.")
        else:
            self.central_model.fresh_model = deepcopy(self.central_model.model)
            self.central_model.distillation = False
        
        #create the mask  
        self.central_model.pruner.create_masks(self.central_model.fresh_model,self.config_params.n_experiences)
        
        #task_mask=self.central_model.pruner.masks

        return
    
    #da fare
    #prendere i path dove le reti vengono salvate 
    #aprire i path
    #metterle dentro a un modello
    #aggregare 
    #salvare la rete

    async def aggregate_models():
        return 
    
    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_text(self, websocket: WebSocket, message: str) -> None:
        await websocket.send_text(message)

    async def send_json(self, websocket: WebSocket, data: dict) -> None:
        #r = json.dumps(data)
        await websocket.send_json(data)

    async def broadcast_text(self, message: str) -> None:
        for ws in self.active_connections:
            await ws.send_text(message)



class ClientMessage(BaseModel):
    status: int
    ask: int
    task: int
    network_position: Optional[str]



async def run_aggregation(networks: list):
    print("Starting aggregator on:", networks)
    #load the networks 

    #load the masks 
    await asyncio.sleep(5)  # Simulate heavy work
    
    print("Aggregator finished")


async def create_masks():
    return

app = FastAPI()
config_parameters=build_server_model()
manager = ConnectionManager(config_parameters)

N_CLIENTS = 1

@app.get("/")
async def root():
    html = Path("static/index.html").read_text(encoding="utf8")
    return HTMLResponse(html)


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):

    await manager.connect(websocket)

    try:
        while True:

            raw_data = await websocket.receive_json()
            
            message = ClientMessage(**raw_data)

            status = message.status
            task_id = message.task

            # --- STATUS 1: client is requesting network or task state ---
            if status == 1:

                # ask == 0 -> client wants aggregator position
                if message.ask == 0:
                    if task_id not in manager.tasks:
                        # Only allow sequential creation of tasks
                        if not manager.tasks or task_id == max(manager.tasks.keys()) + 1:
                            manager.tasks[task_id] = 0
                        else:
                            continue  # invalid task order place and error 

                    response = {"aggregator_pos": "posizione"}

                # ask == 1 -> client asking if task is ready
                else:
                    ready = manager.tasks.get(task_id, None)

                    if ready is None:
                        continue  # invalid task id place an error 

                    response = {"ready": ready}
                
                await manager.send_json(websocket, response)

            # --- STATUS 2: client has finished training and sends network ---
            elif status == 2:

                manager.trained_networks.append(message.network_position)

                await manager.send_json(websocket, {"received": 1})

                # If all clients finished -> run aggregator
                if len(manager.trained_networks) >= N_CLIENTS:

                    await run_aggregation(manager.trained_networks)

                    # Reset
                    manager.trained_networks.clear()
                    manager.tasks[task_id] = 1
                    #await manager.broadcast_text("Aggregator finished!")

            else:
                print("Invalid status received:", status)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast_text(f"Client #{client_id} disconnected")

#if __name__=="__main__":
    
    