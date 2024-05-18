import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from PLayer import PLayer  # Assuming your custom model is defined in PLayer.py
import copy
import onnx
from onnxoptimizer import optimize
from torchsummary import summary


# Instantiate your custom model on CPU
original_model = PLayer(in_channels=3, out_channels=3, n_feat=40, num_blocks=[1, 2, 2])

# Load the pre-trained weights
checkpoint = torch.load('./net_g_16000_1024_40.pth')

original_model.load_state_dict(checkpoint['params'])

# Create a copy of the original model
copied_model = copy.deepcopy(original_model)

for name, module in copied_model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        print(f"Before pruning, Number of non-zero parameters in {name}: {torch.sum(module.weight != 0)}")

# Define a dummy input to initialize the model
dummy_input = torch.randn(1, 3, 512, 512)

# Specify the layers you want to prune in the copied model
# Assuming your model is named 'my_model'
parameters_to_prune = []
# Iterate through the model's named modules
for name, module in copied_model.named_modules():
    # Check if the module is a Conv2d layer
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        # Prune the 'weight' parameter of the Conv2d layer
        parameters_to_prune.append((module, 'weight'))

# Perform model pruning on specified layers in the copied model on CPU
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,  # Example: prune 20% of the weights
)

# Remove pruned weights from the model
for module, parameter in parameters_to_prune:
    prune.remove(module, parameter)

# Print the pruned parameters after pruning removal
for name, module in copied_model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        pruned_params = torch.sum(module.weight != 0)
        print(f"Number of non-zero parameters in {name} after pruning: {pruned_params}")

# Optional: Quantization on CPU
# Quantize the pruned model
# quantized_model = torch.quantization.quantize_dynamic(copied_model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8)

# Run a forward pass on CPU
# out = quantized_model(dummy_input)

# for name, module in copied_model.named_modules():
#     if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
#         print(f"Number of non-zero parameters in {name}: {torch.sum(module.weight != 0)}")

# Save the pruned and quantized model on CPU
# torch.save(quantized_model, "deployment/pruned_quantized_model_organoid_1024_40_02.pth")
torch.save(copied_model, "deployment/pruned_quantized_model_organoid_02.pth")

# Move the model to CPU
copied_model.to('cuda')

# Move the dummy input to CPU
dummy_input_cpu = dummy_input.to('cuda')

summary(copied_model, input_size=(3, 1024, 1024))

# onnx_path = "deployment/pruned_quantized_model_organoid.onnx"
#
# torch.onnx.export(copied_model, dummy_input, onnx_path, verbose=True)
# onnx_model = onnx.load(onnx_path)
# result = optimize(onnx_model)
# optimized_model = result[0] if isinstance(result, tuple) else result
# onnx.save(optimized_model, "deployment/optimized_model.onnx")