import torch
import torchvision.models as models
import numpy as np
import onnxruntime


def load_ScriptModule(path):
    return torch.jit.load(path).eval()

def save_ScriptModule(model, path):
    model.save(path)

def load_model(path):
    return torch.load(path).eval()

def save_model(model, path):
    torch.save(model, path)

def get_model_from_torchvision(model_name, pretrained=True):
    return getattr(models, model_name)(pretrained=pretrained).eval()


def export_onnx(model, model_input, path, verbose=True, input_names=['data'], output_names=['output1'], example_outputs=None):
    torch.onnx.export(model, dummy_input, path, verbose=verbose, input_names=input_names, output_names=output_names, example_outputs=example_outputs)


def get_rand_input(shape):
    return torch.randn(shape)

def get_pt_path(model_name):
    return "../models/pytorch/" + model_name + ".pt"


def get_onnx_path(model_name):
    return "../models/onnx/" + model_name + ".onnx"


def test_pt(dummy_input, model, model_name):
    model1 = load_ScriptModule(get_pt_path(model_name))
    model.eval()
    
    result = model(dummy_input).detach().numpy()
    result1 = model(dummy_input).detach().numpy()

    return np.allclose(result, result1, rtol=1e-05, atol=1e-08)

def test_onnx(dummy_input, model, model_name):
    model.eval()
    result = model(dummy_input).detach().numpy()
    
    session = onnxruntime.InferenceSession(get_onnx_path(model_name))
    result1 = np.array(session.run(["output1"],{"data":dummy_input.detach().numpy()}))

    return np.allclose(result, result1, rtol=1e-05, atol=1e-05)


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
        if "inception" in model_name:
            dummy_input = torch.randn(1, 3, 299, 299)
        else:
            dummy_input = torch.randn(1, 3, 224, 224)

        model = get_model_from_torchvision(model_name)
        model = torch.jit.trace(model, dummy_input)

        print("saving model {}...".format(model_name))
        save_ScriptModule(model, get_pt_path(model_name))
        export_onnx(model, dummy_input, get_onnx_path(model_name), example_outputs=model(dummy_input))
        
        print("Check the result between pytorch and .pt: {}".format(test_pt(dummy_input, model, model_name)))
        print("Check the result between pytorch and .onnx: {}".format(test_onnx(dummy_input, model, model_name)))