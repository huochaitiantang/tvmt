import os
import torch
import torchvision.models as models
import numpy as np
import onnxruntime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None, help='a chosen model, like resnet18_v2', required=False)
parser.add_argument('--verbose', type=bool, default=False, help='model export verbose', required=False)
args = parser.parse_args()


def get_model_from_torchvision(model_name, pretrained=True):
    return getattr(models, model_name)(pretrained=pretrained).eval()


def export_onnx(model, model_input, path, input_names=['data'], output_names=['output1'], example_outputs=None):
    torch.onnx.export(model, dummy_input, path, verbose=args.verbose, input_names=input_names, output_names=output_names, example_outputs=example_outputs)

def get_onnx_dir():
    return "../../models/onnx/"


def get_onnx_path(model_name):
    return get_onnx_dir() + model_name + ".onnx"

def test_onnx(dummy_input, model, model_name):
    model.eval()
    result = model(dummy_input).detach().numpy()
    
    session = onnxruntime.InferenceSession(get_onnx_path(model_name))
    result1 = np.array(session.run(["output1"],{"data":dummy_input.detach().numpy()}))

    return np.allclose(result, result1, rtol=1e-05, atol=1e-05)


if __name__ == "__main__":
    model_name = args.model
    os.makedirs(get_onnx_dir(), exist_ok=True)

    if "inception" in model_name:
        dummy_input = torch.randn(1, 3, 299, 299)
    else:
        dummy_input = torch.randn(1, 3, 224, 224)

    model = get_model_from_torchvision(model_name)
    model = torch.jit.trace(model, dummy_input)

    print("saving model {}...".format(model_name))
    export_onnx(model, dummy_input, get_onnx_path(model_name), example_outputs=model(dummy_input))
    
    print("Check the result between pytorch and .onnx: {}".format(test_onnx(dummy_input, model, model_name)))