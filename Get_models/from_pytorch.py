import torch
import torchvision.models as models
import numpy as np


def load_model(path):
    return torch.load(path)

def save_model(model, path):
    torch.save(model, path)

def get_model_from_torchvision(model_name, pretrained=True):
    return getattr(models, model_name)(pretrained=pretrained)


def export_onnx(model, model_input, path, verbose=True, input_names=['input1'], output_names=['output1']):
    torch.onnx.export(model, dummy_input, path, verbose=verbose, input_names=input_names, output_names=output_names)


def get_rand_input(shape):
    return torch.randn(shape)

def get_pt_path(model_name):
    return "models/pytorch/" + model_name + ".pt"


def get_onnx_path(model_name):
    return "models/onnx/pytorch/pytorch_" + model_name + ".onnx"


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
    
    for model_name in model_names:
        model = get_model_from_torchvision(model_name).eval()
        
        print("saving model {}...".format(model_name))
        save_model(model, get_pt_path(model_name))
        
        if "inception" in model_name:
            dummy_input = torch.randn(1, 3, 299, 299)
        else:
            dummy_input = torch.randn(1, 3, 224, 224)

        export_onnx(model, dummy_input, get_onnx_path(model_name))

        # model1 = load_model(get_pt_path(model_name))
        # model1.eval()
        
        # result = model(dummy_input).detach().numpy()
        # result1 = model1(dummy_input).detach().numpy()
        
        # print(np.allclose(result, result1, rtol=1e-05, atol=1e-08))
