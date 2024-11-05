from distcuda.api import DistCuda

# Initialize with the configuration file
distcuda = DistCuda("path/to/distcuda_config.json")

model = MyModel()
dataloader = DataLoader()

distcuda.train(model, dataloader)