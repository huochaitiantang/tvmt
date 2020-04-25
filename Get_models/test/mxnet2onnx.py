# some standard imports
import mxnet as mx
import numpy as np
import os
import argparse
import sys

from mxnet.gluon.model_zoo.vision import get_model
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None, help='a chosen model, like resnet18_v2', required=False)
args = parser.parse_args()

def getData(path, data_lists):
    with open(path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            data_lists.append(line)

def get_model_names():
    path = '../models/mxnet/model_names'
    model_names = []
    getData(path, model_names)
    return model_names

models = get_model_names()
print(models)

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
    mx.model.save_checkpoint(path+"/"+model_name, 0, mx_sym, args, auxs)
    # there are 'xx.params' and 'xx-symbol.json' on disk


def convert_sym_params_to_onnx(model_name, path_sym_params, path_onnx):
    img_size = 299 if 'inceptionv3' in model_name else 224
    input_shape = (1, 3, img_size, img_size)
    # symbol and params
    sym = path_sym_params + '/' + model_name + '-symbol.json'
    params = path_sym_params + '/' + model_name + '-0000.params'
    # Path of the output file
    onnx_file = path_onnx+'/'+model_name+'.onnx'
    # Invoke export model API. It returns path of the converted onnx model
    converted_model_path = onnx_mxnet.export_model(
        sym, params, [input_shape], np.float32, onnx_file, verbose=True)

def mxnet2onnx(model_name, path_sym_params, path_onnx):
    block = get_model(model_name, pretrained=True)
    save_models(block, model_name, path_sym_params)
    convert_sym_params_to_onnx(model_name, path_sym_params, path_onnx)


def main():
    path_onnx = '../models/onnx/'
    path_sym_params = './symbol_and_params'
    os.makedirs(path_onnx, exist_ok=True)

    model_names = models
    #for model_name in model_names:
    #    print("model name : "+ model_name)
    #    mxnet2onnx(model_name, path_sym_params, path_onnx)
    model_name = args.model
    mxnet2onnx(model_name, path_sym_params, path_onnx)


if __name__ == "__main__":
    main()

