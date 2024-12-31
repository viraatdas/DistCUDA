from distcuda.api import DistCuda

# Initialize with the configuration file
distcuda = DistCuda("path/to/distcuda_config.json")

model = MyModel()
dataloader = DataLoader()

# Train the model

## Simple case
metrics = distcuda.train(model, dataloader)

## With custom config
config = TrainingConfig(batch_size=64, num_epochs=20)
metrics = distcuda.train(model, dataloader, config)

# Inference
predictions = distcuda.predict(model, test_dataloader)

# Check available devices
devices = distcuda.available_devices()

