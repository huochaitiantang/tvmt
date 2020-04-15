import torch
import torchvision.models as models

# Use an existing model from Torchvision, note it 
# will download this if not already on your computer (might take time)
#model = models.alexnet(pretrained=True)
#model = models.resnet18(pretrained=True)

resnet18            = models.resnet18(pretrained=True)
alexnet             = models.alexnet(pretrained=True)
squeezenet1_0       = models.squeezenet1_0(pretrained=True)
vgg16               = models.vgg16(pretrained=True)
densenet161         = models.densenet161(pretrained=True)
inception_v3        = models.inception_v3(pretrained=True)
googlenet           = models.googlenet(pretrained=True)
shufflenet_v2_x1_0  = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet_v2        = models.mobilenet_v2(pretrained=True)
resnext50_32x4d     = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2     = models.wide_resnet50_2(pretrained=True)
mnasnet1_0          = models.mnasnet1_0(pretrained=True)

# Create some sample input in the shape this model expects
dummy_input = torch.randn(1, 3, 224, 224)
dummy_input_inception = torch.randn(1, 3, 299, 299)

# It's optional to label the input and output layers
#input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
input_names = [ "input1" ]
print(input_names)
output_names = [ "output1" ]

# Use the exporter from torch to convert to onnx 
# model (that has the weights and net arch)
#torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
path_onnx = "../onnx/models_onnx/"

torch.onnx.export(resnet18,             dummy_input,            path_onnx + "resnet18.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(alexnet,              dummy_input,            path_onnx + "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(squeezenet1_0,        dummy_input,            path_onnx + "squeezenet1_0.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(vgg16,                dummy_input,            path_onnx + "vgg16.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(densenet161,          dummy_input,            path_onnx + "densenet161.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(inception_v3,         dummy_input_inception,  path_onnx + "inception_v3.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(googlenet,            dummy_input,            path_onnx + "googlenet.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(shufflenet_v2_x1_0,   dummy_input,            path_onnx + "shufflenet_v2_x1_0.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(mobilenet_v2,         dummy_input,            path_onnx + "mobilenet_v2.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(resnext50_32x4d,      dummy_input,            path_onnx + "resnext50_32x4d.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(wide_resnet50_2,      dummy_input,            path_onnx + "wide_resnet50_2.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(mnasnet1_0,           dummy_input,            path_onnx + "mnasnet1_0.onnx", verbose=True, input_names=input_names, output_names=output_names)






