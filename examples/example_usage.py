import distcuda.api

# Initialize with the configuration file
distcuda.api.init('path/to/distcuda_config.json')

@distcuda.task
def my_gpu_task(data):
    # Your CUDA code here
    return data * 2  # Example processing

result = my_gpu_task(10)
print(result)  # Should print 20