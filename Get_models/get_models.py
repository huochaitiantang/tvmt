import os
import sys
import argparse
import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--framework', type=str, default=None, help='a chosen framework, like mxnet, onnx or tensorflow', required=False)
parser.add_argument('--model', type=str, default=None, help='a chosen model, like resnet18_v2', required=False)

args = parser.parse_args()
print(args)

def getData(path, data_lists):
    with open(path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            data_lists.append(line)

def get_model_names():
    path = './models/mxnet/model_names'
    model_names = []
    getData(path, model_names)
    return model_names

models = get_model_names()
print(models)

print( args.framework)
print( args.model)
framework =['mxnet', 'onnx', 'tensorflow', 'pytorch'] 
if args.framework not in framework:
    print( str(args.framework) + " not in " + str(framework) )
    sys.exit()
if args.model not in models:
    print( str(args.model) + " not in " + str(models) )
    sys.exit()


def block2symbol(block):
    data = mx.sym.Variable('data')
    sym = block(data)
    args = {}
    auxs = {}
    for k, v in block.collect_params().items():
        args[k] = mx.nd.array(v.data().asnumpy())
    return sym, args, auxs


def save_models(block, model_name, path):
    mx_sym, args, auxs = block2symbol(block)
    # usually we would save/load it as checkpoint
    os.makedirs(path, exist_ok=True)
    mx.model.save_checkpoint(path+model_name, 0, mx_sym, args, auxs)
    # there are 'xx.params' and 'xx-symbol.json' on disk


def get_models_mxnet(model_name):
    block = get_model(model_name, pretrained=True)
    this_file_path = os.path.dirname(__file__)
    path_sym_params = os.path.join(this_file_path, './models/mxnet/')
    print(path_sym_params)
    save_models(block, model_name, path_sym_params)


def get_models_pytorch(model_name):
    import torch
    import torchvision.models as models

    model = getattr(models, model_name)(pretrained=True).eval()
    if "inception" in model_name:
        dummy_input = torch.randn(1, 3, 299, 299)
    else:
        dummy_input = torch.randn(1, 3, 224, 224)
    
    script = torch.jit.trace(model, dummy_input)
    current_path = os.path.dirname(__file__)
    save_path = os.path.join(current_path, './models/pytorch/')
    onnx_path = os.path.join(current_path, './models/onnx/pytorch/')
    script.save(save_path + model_name + '.pt')

    torch.onnx.export(script, dummy_input, onnx_path + 'pytorch_' + model_name + '.onnx', verbose=True, input_names=['data'], output_names=['output1'], example_outputs=script(dummy_input))

def main():
    if args.framework == 'mxnet':
        get_models_mxnet(args.model)
    elif args.framework == 'tensorflow':
        get_models_onnx(args.model)
    elif args.framework == 'onnx':
        get_models_onnx(args.model)
    elif args.framework == 'pytorch':
        get_models_pytorch(args.model)

if __name__ == '__main__':
    main()

