import os
import sys
import argparse
import tvm.relay as relay
import tvm
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import numpy as np
import tvm.contrib.graph_runtime as runtime

framework =['mxnet', 'onnx', 'tensorflow'] 
target = ['x86', 'gpu', 'arm', 'aarch64']
tuned = ['Yes', 'No']

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default=None, help='a chosen target, like x86, gpu, arm or aarch64', required=False, choices=target)
parser.add_argument('--framework', type=str, default='onnx', help='a chosen framework, like mxnet, onnx or tensorflow', required=False, choices=framework)
parser.add_argument('--model', type=str, default=None, help='a chosen model, like resnet18_v2', required=False)
parser.add_argument('--tuned', type=str, default='No', help='test speed with tuned log or not', required=False, choices=tuned)
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()
print(args)


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
    log_file_path = '../Auto_tune/log/' + args.target + '/' + args.framework + '/'
    log_file = log_file_path + args.target + '_' + args.framework + '_' + model_name +".log"
    print(log_file)
    return log_file


def get_lib_json_params( path ):
    loaded_json = open( path + ".json" ).read()
    loaded_lib = tvm.module.load( path + '.tar' )
    loaded_params = bytearray(open( path + ".params", "rb").read())
    return loaded_json, loaded_lib, loaded_params


def running(graph, lib, path_lib, name_lib, params, input_shape, input_data, input_name, device_key, dtype= 'float32', use_android = False):

    if args.target == 'arm' or args.target == 'aarch64':
        # upload module to device
        print("Upload...")
        remote = autotvm.measure.request_remote(device_key, '0.0.0.0', 9190,
                                                timeout=10000)
        remote.upload(path_lib)
        rlib = remote.load_module(name_lib)

        # upload parameters to device
        target = get_target()
        ctx = remote.context(str(target), 0)
        module = runtime.create(graph, rlib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        #module.set_input(**params)
        module.load_params( params )
        number=1
        repeat=10

    elif args.target == 'x86':
        # upload parameters to device
        ctx = tvm.cpu()
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module = runtime.create(graph, lib, ctx)
        module.set_input(input_name, data_tvm)
        #module.set_input(**params)
        module.load_params( params )
        number=100
        repeat=3

    elif args.target == 'gpu':
        ctx = tvm.gpu()
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module = runtime.create(graph, lib, ctx)
        module.set_input(input_name, data_tvm)
        #module.set_input(**params)
        module.load_params( params )
        number=1
        repeat=600

    # evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=number, repeat=repeat)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
          (np.mean(prof_res), np.std(prof_res)))



def speed ( model_name, tuned = 'No' ):

    dtype = 'float32'
    batch_size = args.batch_size
    input_shape = ( batch_size, 3, 224, 224 )
    if 'inception' in model_name:
        input_shape = ( batch_size, 3, 299, 299 )
    input_data = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    input_name = 'data'
    device_key = None
    if args.target == 'arm':
        device_key = 'rasp3b'
    elif args.target == 'aarch64':
        device_key = 'rk3399'
    elif args.target == 'gpu':
        device_key = 'V100'

    if tuned == 'Yes':
        print( "get tuned lib" )
        deploy_name = args.target + '_' + args.framework + '_' + model_name
        path = '../Auto_tune/lib_json_params/' + args.target + '/' + args.framework + '/'
        path = path + deploy_name 
        graph, lib, params  = get_lib_json_params( path )
        path_lib = path + '.tar' 
        print( "path of lib is : " + path_lib )
        name_lib = deploy_name + '.tar'
        running(graph, lib, path_lib, name_lib, params, input_shape, input_data, input_name, device_key, dtype)
    else:
        print( "get no tuned lib" )
        deploy_name = args.target + '_' + args.framework + '_' + model_name
        path = '../Relay_frontend/lib_json_params/' + args.target + '/' + args.framework + '/'
        path = path + deploy_name 
        graph, lib, params  = get_lib_json_params( path )
        path_lib = path + '.tar' 
        print( "path of lib is : " + path_lib )
        name_lib = deploy_name + '.tar'
        running(graph, lib, path_lib, name_lib, params, input_shape, input_data, input_name, device_key, dtype)


def main():
    model_name = args.model
    speed( model_name, args.tuned )

if __name__ == '__main__':
    main()

