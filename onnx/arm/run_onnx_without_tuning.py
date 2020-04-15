import os

import numpy as np
import tvm
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime

from mxnet.gluon.model_zoo.vision import get_model
import onnx

#################################################################
# Define network
# --------------
# First we need to define the network in relay frontend API.
# We can load some pre-defined network from :code:`relay.testing`.
# We can also load models from MXNet, ONNX and TensorFlow.

def get_network(name, batch_size, dtype, input_name):
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)
    if  'inception' in name:
        input_shape = (1, 3, 299, 299)

    model_path = '../models_onnx/resnet18.onnx'
    onnx_model = onnx.load(model_path)
    #input_name = 'input1'
    print("input_name: "+input_name)
    shape_dict = {input_name: input_shape}
    print(shape_dict)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    #block = get_model(name, pretrained=True)
    #mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
    net = mod["main"]
    net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
    mod = relay.Module.from_expr(net)
    return mod, params, input_shape, output_shape



# Replace "aarch64-linux-gnu" with the correct target of your board.
# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
target = tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu')

# Also replace this with the device key in your tracker
device_key = 'rk3399'

# Set this to True if you use android phone
use_android = False

#### TUNING OPTION ####
dtype = 'float32'

# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=True,
               try_spatial_pack_depthwise=False):
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass

    # if we want to use spatial pack for depthwise convolution
    if try_spatial_pack_depthwise:
        tuner = 'xgb_knob'
        for i in range(len(tasks)):
            if tasks[i].name == 'topi_nn_depthwise_conv2d_nchw':
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host,
                                          'contrib_spatial_pack')
                tasks[i] = tsk

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        n_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt, network, dtype, input_name):
    # extract workloads from relay program
    print("Extract tasks...")
    batch_size = 1
    mod, params, input_shape, _ = get_network(network, batch_size, dtype, input_name)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.nn.conv2d,))

    # get log_filename
    log_file = tuning_opt['log_filename']

    # run tuning tasks
    if os.path.exists(log_file):
        print(log_file + " exists, skipping...")
    else:
        print("Tuning...")
        #tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    #with autotvm.apply_history_best(log_file):
    print("Compile...")
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build_module.build(
            mod, target=target, params=params)

    # export library
    tmp = tempdir()
    if use_android:
        from tvm.contrib import ndk
        filename = "net.so"
        lib.export_library(tmp.relpath(filename), ndk.create_shared)
    else:
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

    # upload module to device
    print("Upload...")
    remote = autotvm.measure.request_remote(device_key, '0.0.0.0', 9190,
                                            timeout=10000)
    remote.upload(tmp.relpath(filename))
    rlib = remote.load_module(filename)

    # upload parameters to device
    ctx = remote.context(str(target), 0)
    module = runtime.create(graph, rlib, ctx)

    #dtype = 'float32'
    #tvm_output = intrp.evaluate()(tvm.nd.array(x.astype(dtype)), **params).asnumpy()
    #
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    #module.set_input('input1', data_tvm)
    module.set_input(input_name, data_tvm)
    module.set_input(**params)

    # evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
          (np.mean(prof_res), np.std(prof_res)))
    

   
# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.


import time
def main(model_names):
    batch_size = 1
    dtype = "float32"
    # Set the input name of the graph
    # For ONNX models, it is typically "0".
    input_name = "input1"

    for model_name in model_names:
        print("model_name : "+model_name)
        print("sleeping...")
        #time.sleep(1*10*60)
        network = model_name
        log_file_path = './log_file/'
        log_file = log_file_path + "%s.%s.log" % (device_key, network)
        print(log_file)

        tuning_option = {
            'log_filename': log_file,
        
            'tuner': 'xgb',
            'n_trial': 1,
            'early_stopping': 1,
        
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

        tune_and_evaluate(tuning_option, network, dtype, input_name)

if __name__ == '__main__':
    model_names = [
        'resnet18',          
        'alexnet',           
        'squeezenet1_0',     
        'vgg16',             
        'densenet161',       
        'inception_v3',      
        'googlenet',         
        'shufflenet_v2_x1_0',
        'mobilenet_v2',      
        'resnext50_32x4d',   
        'wide_resnet50_2',   
        'mnasnet1_0',        
    ]
    #model_names = [
    #    'inception_v3',      
    #    ]

    main(model_names)


