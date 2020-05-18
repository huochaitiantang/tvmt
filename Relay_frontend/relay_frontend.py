import os
import sys
import argparse
import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model
from tvm.contrib import util
import tvm.relay as relay
import tvm


framework =['mxnet', 'onnx', 'tensorflow'] 
target = ['x86', 'gpu', 'arm', 'aarch64']

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default=None, help='a chosen target, like x86, gpu, arm or aarch64', required=False, choices=target)
parser.add_argument('--framework', type=str, default=None, help='a chosen framework, like mxnet, onnx or tensorflow', required=False, choices=framework)
parser.add_argument('--model', type=str, default=None, help='a chosen model, like resnet18_v2', required=False)
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()
print(args)


def get_models_mxnet(model_name, shape_dict):
    path = '../Get_models/models/mxnet/'
    # for a normal mxnet model, we start from here
    #mx_sym, args, auxs = mx.model.load_checkpoint('resnet18_v1', 0)
    mx_sym, args, auxs = mx.model.load_checkpoint( path + model_name , 0 )
    # now we use the same API to get Relay computation graph
    mod, relay_params = relay.frontend.from_mxnet(mx_sym, shape_dict,
                                              arg_params=args, aux_params=auxs)
    return mod , relay_params

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



def relay_save_lib(model_name, mod, params):
    ## we want a probability so add a softmax operator
    func = mod["main"]
    func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)

    target = get_target()

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(func, target, params=params)

    # save the graph, lib and params into separate files
    deploy_name = args.target + '_' + args.framework + '_' + model_name
    path = './lib_json_params/' + args.target + '/' + args.framework + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    lib.export_library( path + deploy_name + '.tar' )

    with open( path + deploy_name + ".json", "w") as fo:
        fo.write(graph)
    with open( path + deploy_name + ".params", "wb") as fo:
        fo.write(relay.save_param_dict(params))


def relay_save_lib_mxnet(model_name):
    input_shape = (args.batch_size, 3, 224, 224 )
    if 'inception' in model_name:
        input_shape = (args.batch_size, 3, 299, 299 )

    shape_dict = {'data': input_shape}
    mod, params = get_models_mxnet(model_name, shape_dict)
    relay_save_lib(model_name, mod, params)


def get_models_onnx(model_name, shape_dict):
    import onnx

    path = '../Get_models/models/onnx/'
    model = onnx.load(path + model_name + '.onnx')
    mod, relay_params = relay.frontend.from_onnx(model, shape_dict)

    return mod, relay_params

def relay_save_lib_onnx(model_name):
    input_shape = (args.batch_size, 3, 224, 224)
    if 'inception' in model_name:
        input_shape = (args.batch_size, 3, 299, 299)
    
    shape_dict = {'data': input_shape}
    mod, params = get_models_onnx(model_name, shape_dict)
    relay_save_lib(model_name, mod, params)


def get_models_tensorflow(model_name, shape_dict):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
    layout = "NCHW"
    model_path = '../Get_models/models/tensorflow/'
    model_path = model_path + model_name + '.pb'
    #graph = tf.get_default_graph()
    #graph_def = graph.as_graph_def()
    #graph_def.ParseFromString(gfile.FastGFile(model_path, 'rb').read())
    
    with tf.compat.v1.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        # Call the utility to import the graph definition into default graph.
        #graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        # Add shapes to the graph.
        #with tf.compat.v1.Session() as sess:
        #    graph_def = tf_testing.AddShapesToGraphDef(sess, 'softmax')
    
    mod, relays_params = relay.frontend.from_tensorflow(graph_def,
                                             layout=layout,shape=shape_dict)
    return mod, relays_params

def relay_save_lib_tensorflow(model_name):
    if 'inception' in model_name:
        input_shape = (args.batch_size, 224, 224, 3)
        shape_dict = {'input':input_shape}
    else:
        input_shape = (args.batch_size, 224, 224, 3)
        shape_dict = {'input':input_shape}
    mod, params = get_models_tensorflow(model_name, shape_dict)
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

