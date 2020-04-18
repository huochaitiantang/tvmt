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
    temp = util.tempdir()
    with open(temp.relpath(graph_path), 'w') as f:
        f.write(graph)
    
    lib.export(lib_path)

    with open(temp.relpath(params_path), 'wb') as f:
        f.write(relay.save_param_dict(params))


def get_module_path(hardware, framework, model_name):
    return '../lib_json_params/' + hardware + '/' + '_'.join([hardware, framework, model])


def file_walk(path):
    return os.listdir(path)

def onnx_convert_relay(onnx_model, shape_dict, target):
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)
    
    return graph, lib, params
