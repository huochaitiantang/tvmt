import torch
import torchvision.models as models

# Use an existing model from Torchvision, note it 
# will download this if not already on your computer (might take time)
#model = models.alexnet(pretrained=True)
model = models.resnet18(pretrained=True)

# Create some sample input in the shape this model expects
dummy_input = torch.randn(1, 3, 224, 224)

# It's optional to label the input and output layers
#input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
input_names = [ "input1" ]
print(input_names)
output_names = [ "output1" ]

# Use the exporter from torch to convert to onnx 
# model (that has the weights and net arch)
#torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
path_onnx = "../onnx/models_onnx/"
torch.onnx.export(model, dummy_input, path_onnx + "resnet18.onnx", verbose=True, input_names=input_names, output_names=output_names)






