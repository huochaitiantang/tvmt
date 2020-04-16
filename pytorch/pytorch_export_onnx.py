import torch
import torchvision.models as models

# Create some sample input in the shape this model expects
def rand_input(model_name, batch_size=1, channels_number=3):
    input_size = 224
    if 'inception' in model_name:
        input_size = 299

    return torch.randn(batch_size, channels_number, input_size, input_size)

# Use an existing model from Torchvision, note it 
# will download this if not already on your computer (might take time)
def get_model(model_name, pretrained=True):
    return getattr(models, model_name)(pretrained=pretrained).eval()

# Use the exporter from torch to convert to onnx 
# model (that has the weights and net arch)
def export_onnx(model, model_input, path, verbose=True, input_names=['input1'], output_names=['output1']):
    torch.onnx.export(model, dummy_input, path, verbose=verbose, input_names=input_names, output_names=output_names)

if __name__ == "__main__":
    model_names = ["resnet18",
                    "alexnet",
                    "squeezenet1_0",
                    "vgg16",
                    "densenet161",
                    "inception_v3",
                    "googlenet",
                    "shufflenet_v2_x1_0",
                    "mobilenet_v2",
                    "resnext50_32x4d",
                    "wide_resnet50_2",
                    "mnasnet1_0"]

    path_onnx = "../onnx/models_onnx/"

    for model_name in model_names:
        model = get_model(model_name)
        dummy_input = rand_input(model_name)
        
        print("exporting {}...".format(model_name))
        export_onnx(model, dummy_input, path_onnx + model_name + '.onnx')

