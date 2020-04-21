import os
import sys
import argparse
import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model
from tvm.contrib import util
import tvm.relay as relay
import tvm
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default=None, help='a chosen target, like x86, gpu, arm or aarch64', required=False)
parser.add_argument('--framework', type=str, default=None, help='a chosen framework, like mxnet, onnx or tensorflow', required=False)
parser.add_argument('--model', type=str, default=None, help='a chosen model, like resnet18_v2', required=False)
parser.add_argument('--tuned', type=str, default=None, help='test speed with tuned log or not', required=False)

args = parser.parse_args()
print(args)


def getData(path, data_lists):
    with open(path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            data_lists.append(line)

def get_model_names():
    path = '../../Get_models/models/mxnet/model_names'
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
tuned = ['Yes', 'No']

if args.target not in target:
    print( str(args.target) + " not in " + str(target) )
    sys.exit()
if args.framework not in framework:
    print( str(args.framework) + " not in " + str(framework) )
    sys.exit()
if args.model not in models:
    print( str(args.model) + " not in " + str(models) )
    sys.exit()
if args.tuned not in tuned:
    print( str(args.tuned) + " not in " + str(tuned) )
    sys.exit()


def get_target():
    # now compile the graph
    if args.target == 'x86':
        target = tvm.target.create('llvm')
    elif args.target == 'gpu':
        target = tvm.target.cuda()
    elif args.target == 'arm':
        target = tvm.target.create('llvm -device=arm_cpu -target=armv7l-linux-gnueabihf -mattr=+neon')
    elif args.target == 'aarch64':
        target = tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu -mattr=+neon')
    return target


def get_log_file(model_name):
    print("model_name : "+model_name)
    log_file_path = '../../Auto_tune/log/' + args.target + '/' + args.framework + '/'
    log_file = log_file_path + args.target + '_' + args.framework + '_' + model_name +".log"
    print(log_file)
    return log_file


def tuning_mxnet(model_name):
    batch_size = 1
    dtype = "float32"
    # Set the input name of the graph
    # For ONNX models, it is typically "input1".
    input_name = "data"
    if args.target == 'arm':
        device_key = 'rasp3b'
    elif args.target == 'aarch64':
        device_key = 'rk3399'
    use_android = False

    log_file = get_log_file(model_name)

    other_option = {
        'model_name' : model_name,
        'batch_size' : batch_size,
        'dtype' : dtype,
        'input_name' : input_name,
        'device_key' : device_key,
        'use_android' : use_android
    }

    tuning_option = {
        'log_filename': log_file,
        'tuner': 'xgb',
        'n_trial': 1,
        'early_stopping': 150,
        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(
                build_func='ndk' if use_android else 'default'),
            runner=autotvm.RPCRunner(
                device_key, host='0.0.0.0', port=9190,
                number=5,
                timeout=10,
            ),
        ),
    }

    tuning( tuning_option, **other_option )


def get_lib_json_params( path ):
    loaded_json = open( path + ".json" ).read()
    loaded_lib = tvm.module.load( path + '.tar' )
    loaded_params = bytearray(open( path + ".params", "rb").read())
    return loaded_json, loaded_lib, loaded_params

def running(graph, lib, path_lib, name_lib, params, input_shape, input_data, input_name, device_key, dtype= 'float32', use_android = False):
    # export library
    #from tvm.contrib.util import tempdir
    #tmp = tempdir()
    #if use_android:
    #    from tvm.contrib import ndk
    #    filename = "net.so"
    #    lib.export_library(tmp.relpath(filename), ndk.create_shared)
    #else:
    #    filename = "net.tar"
    #    lib.export_library(tmp.relpath(filename))

    # upload module to device
    print("Upload...")
    remote = autotvm.measure.request_remote(device_key, '0.0.0.0', 9190,
                                            timeout=10000)
    #remote.upload(tmp.relpath(filename))
    #rlib = remote.load_module(filename)
    remote.upload(path_lib)
    rlib = remote.load_module(name_lib)

    # upload parameters to device
    target = get_target()
    ctx = remote.context(str(target), 0)
    import tvm.contrib.graph_runtime as runtime
    module = runtime.create(graph, rlib, ctx)
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input('data', data_tvm)
    #module.set_input(**params)
    module.set_input(params)

    # evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
          (np.mean(prof_res), np.std(prof_res)))



def speed ( model_name, tuned = 'No' ):

    dtype = 'float32'
    batch_size = 1
    input_shape = ( batch_size, 3, 224, 224 )
    if 'inception' in model_name:
        input_shape = ( batch_size, 3, 299, 299 )
    input_data = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    input_name = 'data'
    if args.target == 'arm':
        device_key = 'rasp3b'
    elif args.target == 'aarch64':
        device_key = 'rk3399'

    if tuned :
        deploy_name = args.target + '_' + args.framework + '_' + model_name
        path = '../../Auto_tune/lib_json_params/' + args.target + '/' + args.framework + '/'
        path = path + deploy_name 
        graph, lib, params  = get_lib_json_params( path )
        path_lib = path + '.tar' 
        name_lib = deploy_name + '.tar'
        running(graph, lib, path_lib, name_lib, params, input_shape, input_data, input_name, device_key, dtype)
    else:
        deploy_name = args.target + '_' + args.framework + '_' + model_name
        path = '../../Relay_frontend/lib_json_params/' + args.target + '/' + args.framework + '/'
        path = path + deploy_name 
        graph, lib, params  = get_lib_json_params( path )
        path_lib = path + '.tar' 
        name_lib = deploy_name + '.tar'
        running(graph, lib, path_lib, name_lib, params, input_shape, input_data, input_name, device_key, dtype)


def main():
    model_name = args.model
    speed( model_name, args.tuned )

if __name__ == '__main__':
    main()

