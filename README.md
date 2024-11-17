# DistCUDA

![Python Test Workflow](https://github.com/viraatdas/DistCUDA/actions/workflows/python-test.yml/badge.svg) - _currently not setup_

DistCUDA allows Python developers to easily distribute and manage training or inference across multiple remote GPUs.

By abstracting the complexities of network communication and task distribution, DistCUDA makes high-performance computing accessible to projects of all sizes.

All you have to do is specify the network address of the machines that have GPUs, and then you can use the `distcuda` API which will seemlessly utilize all the GPUs specified. 

## Features

- [ ] **Automatic GPU Discovery**: Automatically detects and registers available GPUs on the specified network
- [ ] **Flexible Task Distribution**: Distributes computing tasks across multiple remote GPUs based on predefined rules or dynamic scheduling.
- [ ] **Seamless Integration**: Works with existing Python codebases with minimal changes required, making it easy to adopt and integrate.
- [ ] **Scalable and Efficient**: Designed for high scalability and efficiency, optimizing the usage of available GPU resources across the network.

## Installation

_to be filled_

## Quick Start

1. **Configure your GPUs**: First, create a `distcuda_config.json` file to specify the GPUs available in your network. See the [Configuration](#configuration) section for more details.
2. **Initialize DistCUDA**

```python
import distcuda

# Automatically register GPUs based on the config file
distcuda.init('path/to/distcuda_config.json')
```

3. **Run your task**:

```python
@distcuda.task
def my_gpu_task(data):
    # Your CUDA code here
    return processed_data

result = my_gpu_task(data)
```

## Configuration

Create a `distcuda_config.json` file with the following structure:

```json
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
```

Replace the example content with the actual details of the GPUs you wish to use.

## License

DistCUDA is open-source software licensed under the MIT license. See the [LICENSE](LICENSE) file for more details.
