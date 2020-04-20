import os
import onnx
import numpy as np
import tvm
import tvm.relay as relay


# load onnx model
def load_model(path):
    return onnx.load(path)

# save graph, lib, params from relativate path
def save_module(graph, graph_path, lib, lib_path, params, params_path):
    with open(graph_path, 'w') as f:
        f.write(graph)
    
    lib.export_library(lib_path)

    with open(params_path, 'wb') as f:
        f.write(relay.save_param_dict(params))

def save_prefix_module(graph, lib, params, prefix):
    save_module(graph, prefix + '.json', lib, prefix + '.tar', params, prefix + '.params')


def get_module_path(hardware, framework, model_name):
    return '../lib_json_params/' + hardware + '/' + '_'.join([hardware, framework, model_name])


def file_walk(path):
    return os.listdir(path)

def onnx_convert_relay(onnx_model, shape_dict, target):
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)
    
    return graph, lib, params


if __name__ == '__main__':
    hardware_list = ['x86', 'gpu']
    # hardware_list = ['gpu']

    hardware2target = {'x86': 'llvm', 'gpu' : 'cuda'}
    root_path = '../../Get_models/models/onnx/'
    
    framework_list = file_walk(root_path)
    for framework in framework_list:
        framework_path = root_path + framework + '/'
        file_list = file_walk(framework_path)
        # file_list = ['pytorch_shufflenet_v2_x1_0.onnx']

        for file in file_list:
            model = load_model(framework_path + file)

            file = file.replace('.onnx', '')
            model_name = '_'.join(file.split('_')[1:])

            if "inception" in model_name:
                shape_dict = {'data': (1, 3, 299, 299)}
            else:
                shape_dict = {'data': (1, 3, 224, 224)}
            
            for hardware in hardware_list:
                target = hardware2target[hardware]

                module_path = get_module_path(hardware, framework + '2onnx', model_name)
                try:
                    print("begin building")
                    graph, lib, params = onnx_convert_relay(model, shape_dict, target)
                    print("building finished")
                    save_prefix_module(graph, lib, params, module_path)
                    print(model_name)
                except:
                    if os.path.exists(module_path + '.json'):
                        os.rm(module_path + '.json')
                    print("{} failed".format(model_name))

                # from tvm.contrib import graph_runtime
                # m = graph_runtime.create(graph, lib, tvm.cpu())
                # m.set_input('data', tvm.nd.array(np.random.uniform(size=shape_dict['data']).astype('float32')))
                # m.set_input(**params)
                # m.run() 
        