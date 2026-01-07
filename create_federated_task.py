from utils.configuration_utils import create_config
from argparse import ArgumentParser
from split_dataset import run_split
def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("nc", type=int, help=" number of clients for the federated learning setting ")
    arg_parser.add_argument("config_path", type=str, help="path in which store the configurations")

    args = arg_parser.parse_args()


    #create condiguratin file for each node  
    for i in range(args.nc+1):
        
        if i==0:
            filename="server.yaml"
            dict={
                "number_clients":args.nc
            }
            create_config(args.config_path, filename,dict )
        else:
            filename=f"client_{i}.yaml"
            dict={
                "client_id": i
            }

            create_config(f"{args.config_path}/clients", filename,dict )
    
    #run the split of the dataset 
    run_split("CIFAR100", "data_split", "data", args.nc)


if __name__ == "__main__":
    main()