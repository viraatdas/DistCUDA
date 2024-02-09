import functools
from .scheduler import Scheduler

# Global scheduler instance
scheduler = None


def init(config_path):
    """
    Initialize DistCUDA with a configuration file specifying available GPUs.

    Example configuration file:
    {
        "gpus": [
            {
                "id": "gpu_local_1",
                "model": "NVIDIA RTX 3080",
                "networkAddress": "localhost"
            },
            {
                "id": "gpu_remote_1",
                "model": "NVIDIA Tesla V100",
                "networkAddress": "192.168.1.100"
            }
        ]
    }
    
    Args:
        config_path (str): Path to the JSON configuration file.
    """
    global scheduler
    
    # Load the configuration from a JSON file
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    # Initialize the scheduler with the loaded configuration
    scheduler = Scheduler(config['gpus'])

def task(func):
    """
    Decorator to mark a function as a distributable task.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if scheduler is None:
            raise RuntimeError(
                "DistCUDA is not initialized. Call distcuda.init(config_path) before submitting tasks.")

        # Serialize arguments and submit the task to the scheduler
        task_id = scheduler.submit(func, args, kwargs)

        # Wait for the task to complete and return the result
        result = scheduler.wait_for_result(task_id)
        return result

    return wrapper
