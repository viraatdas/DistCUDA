import functools
import json
import paramiko
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable


CONNECTION_TYPES = ["ssh"]

class Device:
    def __init__(self, name, description, network_address):
        """
        Assumption: the GPUs in the machine are all going to be of the same type. 
        So for example, this device might have 8 GPUs. This code assumes that all of the 
        GPUs are of either CUDA or something like that. 

        TODO: address this assumption in the future to allow for a machine having 
        different kinds of GPUs
        """
        self.name = name
        self.description = description
        self.network_address = network_address
        
        self.num_gpus = -1
        self.average_latency = -1 # todo: implement for scheduling purpose
    
    def register(self, connection_type: str = "ssh", gpu_type: str = "cuda"):
        """
        Register the GPU with DistCUDA.

        Returns:
            bool: True if registration was successful, False otherwise.
        """
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh_client.connect(self.network_address)
        except Exception as e:
            print(f"Failed to connect to {self.name}: {e}")
            return False

        # Execute a command on the remote machine to retrieve the number of GPUs
        stdin, stdout, stderr = ssh_client.exec_command("nvidia-smi --list-gpus")
        output = stdout.read().decode()

        # Parse the output to get the number of GPUs
        gpu_count_lines = output.splitlines()
        self.num_gpus = len(gpu_count_lines) - 1  # Subtract 1 for the header line
        
        # test lightweight tensor to see if the GPUs are all working
        # Establish an SSH connection to each GPU and test a lightweight tensor
        for i in range(self.num_gpus):
            gpu_id = f"cuda:{i}"

            try:
                # Test a lightweight tensor on this GPU
                device = torch.device(gpu_id)
                x = torch.randn(1, 100, device=device)

                # Perform some simple operation to test the tensor
                y = x.sum()

                assert x is not None, "Tensor is None on this GPU"
                assert isinstance(y, torch.Tensor), "Result is not a tensor on this GPU"
            except Exception as e:
                print(f"Failed to test tensor on GPU {i}: {e}")
                return False


        ssh_client.close()

        return True


class DistCuda:
    def __init__(self, config_path):
        """
        Initialize DistCUDA with a configuration file specifying available GPUs.

        Example configuration file:
        {
            "devices": [
                {
                    "name": "gpu_local_1",
                    "description": "NVIDIA RTX 3080",
                    "network_address": "localhost",
                    "connection_type": "ssh",
                    "permission_key": "",
                },
                {
                    "name": "gpu_remote_1",
                    "description": "NVIDIA Tesla V100",
                    "network_address": "192.168.1.100",
                    "connection_type": "ssh",
                    "permission_key": "",
                }
            ]
        }
        
        Args:
            config_path (str): Path to the JSON configuration file.
        """

        # Load the configuration from a JSON file
        with open(config_path, 'r') as file:
            config = json.load(file)
        
        self.valid_devices = []
        devices = config["devices"]
        

        for d in devices:
            device = Device(name=d.name, 
                            description=d.description, 
                            network_address=d.network_address)

            success = device.register()
            if success:
                self.valid_devices.append(device)
        
    
    def train(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer = None,
        criterion: Callable = None,
        num_epochs: int = 10,
        batch_size: int = 32
    ) -> None:
        """Train the model using distributed training.

        Args:
            model (nn.Module): The PyTorch model to be trained.
            dataloader (DataLoader): The data loader for loading the dataset.
            optimizer (Optimizer, optional): The optimizer to use. Defaults to None.
            criterion (Callable, optional): The loss function to use. Defaults to None.
            num_epochs (int, optional): The number of epochs to train for. Defaults to 10.
            device (str, optional): The device to use for training (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
            batch_size (int, optional): The batch size to use during training. Defaults to 32.

        Returns:
            None
        """
        chunk_size = len(dataloader.dataset) // self.num_gpus
        chunks = []
        for i in range(self.num_gpus):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < self.num_gpus - 1 else len(dataloader.dataset)
            chunks.append((start_idx, end_idx))

        # Initialize a list to store the local models and optimizers for each GPU
        local_models = []
        local_optimizers = []
        

        




    