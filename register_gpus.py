import json

def register_gpus_from_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    for gpu in config['gpus']:
        register_gpu(gpu)

def register_gpu(gpu_info):
    # Here, implement the logic to register the GPU with your system.
    # This might involve adding the GPU to a central database, contacting a scheduler,
    # or simply storing the information in a global state within the library.
    print(f"Registering GPU: {gpu_info['id']} at {gpu_info['networkAddress']}")
    # Add more logic as needed based on your system's requirements.
