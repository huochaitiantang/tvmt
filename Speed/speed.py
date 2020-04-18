import os
import tvm
from tvm import relay
from tvm.contrib import util
import tvm.contrib.graph_runtime as runtime

import numpy as np

use_android=False


hardware2target = {
        'x86': 'llvm',
        'gpu': 'cuda',
        'arm': tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu')
        }

hardware2ctx = {
        'x86': tvm.cpu(),
        'gpu': tvm.gpu(1),
        'arm': tvm.context(str(hardware2target['arm']), 0)
}

# get the files and dictionary under path
def file_walk(path):
    return os.listdir(path)

# get path of file under the path
def path_walk(path):
    sub_path = []
    dir_list = file_walk(path)
    for dir_name in dir_list:
        sub_path.append(path + dir_name)
    
    return sub_path


# get all files and dictionary under path
def rfile_walk(path):
    files = []
    for _, dirs, rfiles in os.walk(path):
        files = files + rfiles
    
    return files

# get the relativate path to the .json, .lib and .params
def get_path(model_name, hardware, framework, tuned=False):
    path = '../'
    if tuned:
        path = path + 'Auto_tune/'
    else:
        path = path + 'Relay_frontend/lib_json_params/'
    
    path = path + hardware + '/' + framework + '/'
    name = hardware + '_' + framework + '_' + model_name
    
    return path + name

# get config from the filename. e.g. ../Relay_frontend/gpu/mxnet/gpu_mxnet_resnet18(.lib)
def get_model_config(filename):
    config_name = filename.split('/')[-1]
    if '.' in config_name:
        config_name = '.'.join(config_name.split('.')[:-1])
    config = config_name.split('_')
    
    hardware = config[0]
    framework = config[1]
    model_name = '_'.join(config[2:])
    
    return hardware2target[hardware], framework, model_name

# load graph, lib, params from relativate path
def load_module(graph_path, lib_path, params_path):
    loaded_json = open(graph_path).read()
    loaded_lib = tvm.module.load(lib_path)
    loaded_params = bytearray(open(params_path, "rb").read())

    return loaded_json, loaded_lib, loaded_params

# save graph, lib, params from relativate path
def save_module(graph, graph_path, lib, lib_path, params, params_path):
    with open(graph_path, 'w') as f:
        f.write(graph)
    
    lib.export_library(lib_path)

    with open(params_path, 'wb') as f:
        f.write(relay.save_param_dict(params))


def build_arm(target, device_key, graph, lib):
    tmp = util.tempdir()
    if use_android:
        from tvm.contrib import ndk
        filename = 'net.so'
        lib.export_library(tmp.relpath(filename), ndk.create_shared)
    else:
        filename = 'net.tar'
        lib.export_library(tmp.relpath(filename))

    from tvm import autotvm

    remote = autotvm.measure.request_remote(device_key, '0.0.0.0', 9190, timeout=10000)
    remote.upload(tmp.relpath(filename))
    rlib = remote.load_module(filename)

    ctx = remote.context(str(target), 0)
    module = runtime.create(graph, rlib, ctx)
    
    return module

def build_cpu(graph, lib):
    ctx = tvm.cpu()
    module = runtime.create(graph, lib, ctx)
    
    return module


def build_gpu(graph, lib):
    ctx = tvm.gpu()
    module = runtime.create(graph, lib, ctx)

    return module


def build(target, graph, lib, device_key='rk3399'):
    if str(target) == 'llvm':
        module = build_cpu(graph, lib)
    elif str(target) == 'cuda':
        module = build_gpu(graph, lib)
    else:
        if not device_key:
            print("lost device_key for build module of target {}".format(str(target)))
        module = build_arm(target, device_key, graph, lib)
    
    return module


def evaluate(module, params, input_shape, ctx, dtype='float32', repeat=600):
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))

    module.set_input('data', data_tvm)
    if isinstance(params, bytearray):
        module.load_params(params)
    else:
        module.set_input(**params)

    ftimer = module.module.time_evaluator('run', ctx, number=1,repeat=repeat)
    prof_res = np.array(ftimer().results) * 1000 * 1000

    return np.mean(prof_res), np.std(prof_res)


if __name__ == '__main__':
    root_path = '../Relay_frontend/lib_json_params/'
    batch_size = 1
    file_path = []
    hardware_path_list = path_walk(root_path)
    time_list = {}

    for hardware_path in hardware_path_list:
        hardware = hardware_path.split('/')[-1]
        ctx = hardware2ctx[hardware]

        temp = set()
        for file_path in file_walk(hardware_path):
            temp.add(file_path.split('.')[0])
        
        # path is ../Relay_frontend/x86/x86_mxnet_resnet18
        for path in temp:
            target, framework, model_name = get_model_config(path)
            
            mod_path = hardware_path + '/' + path
            graph, lib, params = load_module(mod_path + '.json', mod_path + '.tar', mod_path + '.params')
            module = build(target, graph, lib)

            if 'inception' in model_name:
                input_shape = (batch_size, 3, 299, 299)    
            else:
                input_shape = (batch_size, 3, 224, 224)

            key = '_'.join([hardware, framework, model_name])
            mean_time, std_time = evaluate(module, params, input_shape, ctx)
            time_list[key] = [mean_time, std_time]
            print('model: {}, time: {:.2f} us({:.2f} us)'.format(key, mean_time, std_time))
            # print('model: {}, time: {:.2f} us({:.2f} us)'.format(key, time_list[key][0], time_list[key][1]))
    
    print('-------------------------------------------')
    print("model \t time(us)")
    for key, value in time_list.items():
        model_name = key
        print("{} \t {:.2f}({:.2f})".format(model_name, value[0], value[1]))
