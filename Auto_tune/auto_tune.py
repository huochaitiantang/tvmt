import os
import sys
import argparse
import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model
from tvm.contrib import util
import tvm.relay as relay
import tvm

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default=None, help='a chosen target, like x86, gpu, arm or aarch64', required=False)
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
    path = '../Get_models/models/mxnet/model_names'
    model_names = []
    getData(path, model_names)
    return model_names

models = get_model_names()
print(models)

print( args.target)
print( args.framework)
print( args.model)
framework =['mxnet', 'onnx', 'tensorflow'] 
target = ['x86', 'gpu', 'arm', 'aarch64']

if args.target not in target:
    print( str(args.target) + " not in " + str(target) )
    sys.exit()
if args.framework not in framework:
    print( str(args.framework) + " not in " + str(framework) )
    sys.exit()
if args.model not in models:
    print( str(args.model) + " not in " + str(models) )
    sys.exit()


def get_models_mxnet(model_name, shape_dict):
    path = '../Get_models/models/mxnet/'
    # for a normal mxnet model, we start from here
    #mx_sym, args, auxs = mx.model.load_checkpoint('resnet18_v1', 0)
    mx_sym, args, auxs = mx.model.load_checkpoint( path + model_name , 0 )
    # now we use the same API to get Relay computation graph
    mod, relay_params = relay.frontend.from_mxnet(mx_sym, shape_dict,
                                              arg_params=args, aux_params=auxs)
    return mod , relay_params

def relay_save_lib(model_name, mod, params):
    ## we want a probability so add a softmax operator
    func = mod["main"]
    func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)

    # now compile the graph
    if args.target == 'x86':
        target = tvm.target.create('llvm')
    elif args.target == 'gpu':
        target = tvm.target.cuda()
    elif args.target == 'arm':
        target = tvm.target.create('llvm -device=arm_cpu -target=armv7l-linux-gnueabihf -mattr=+neon')
    elif args.target == 'aarch64':
        target = tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu -mattr=+neon')

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(func, target, params=params)

    # save the graph, lib and params into separate files
    deploy_name = args.target + '_' + args.framework + '_' + model_name
    path = './lib_json_params/' + args.target + '/' + args.framework + '/'
    lib.export_library( path + deploy_name + '.tar' )

    with open( path + deploy_name + ".json", "w") as fo:
        fo.write(graph)
    with open( path + deploy_name + ".params", "wb") as fo:
        fo.write(relay.save_param_dict(params))


def relay_save_lib_mxnet(model_name):
    input_shape = ( 1, 3, 224, 224 )
    if 'inception' in model_name:
        input_shape = ( 1, 3, 299, 299 )

    shape_dict = {'data': input_shape}
    mod, params = get_models_mxnet(model_name, shape_dict)
    relay_save_lib(model_name, mod, params)


def main():
    model_name = args.model
    if args.framework == 'mxnet':
        relay_save_lib_mxnet(model_name)
    elif args.framework == 'tensorflow':
        relay_save_lib_tensorflow(model_name)
    elif args.framework == 'onnx':
        relay_save_lib_onnx(model_name)

if __name__ == '__main__':
    main()

