# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Auto-tuning a convolutional network for NVIDIA GPU
==================================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, `Eddie Yan <https://github.com/eqy/>`_

Auto-tuning for specific devices and workloads is critical for getting the
best performance. This is a tutorial on how to tune a whole convolutional
network for NVIDIA GPU.

The operator implementation for NVIDIA GPU in TVM is written in template form.
The template has many tunable knobs (tile factor, unrolling, etc).
We will tune all convolution and depthwise convolution operators
in the neural network. After tuning, we produce a log file which stores
the best knob values for all required operators. When the TVM compiler compiles
these operators, it will query this log file to get the best knob values.

We also released pre-tuned parameters for some NVIDIA GPUs. You can go to
`NVIDIA GPU Benchmark <https://github.com/apache/incubator-tvm/wiki/Benchmark#nvidia-gpu>`_
to see the results.
"""

######################################################################
# Install dependencies
# --------------------
# To use the autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado
#
# To make TVM run faster during tuning, it is recommended to use cython
# as FFI of tvm. In the root directory of tvm, execute:
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

import os

import numpy as np

import tvm
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime


import argparse

config = {
    'model_names_file':'model_names.json',
    'exec_time_info':'./log/tuning_exec_info',
}

parser = argparse.ArgumentParser()
for key, value in config.items():
    parser.add_argument('-' + key, default=value)

args = parser.parse_args()
config["model_names_file"] = args.model_names_file
config["exec_time_info"] = args.exec_time_info

dtype = 'float32'
target = tvm.target.cuda()

def get_network(name):
    """Get the symbol definition and random weight of a network"""
    if name == 'inceptionv3':
        input_shape = (1, 3, 299, 299)
    else:
        input_shape = (1, 3, 224, 224)
    output_shape = (1, 1000)

    from mxnet.gluon.model_zoo.vision import get_model
    block = get_model(name, pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
    net = mod["main"]
    net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
    mod = relay.Module.from_expr(net)
    
    return mod, params, input_shape, output_shape


###################################################################
# Begin Tuning
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=True):
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

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
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

def tune_and_evaluate(tuning_opt, network):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, out_shape = get_network(network)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params, ops=(relay.op.nn.conv2d,))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # export library
        tmp = tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        # load parameters
        # ctx = tvm.context(str(target), 0)
        ctx = tvm.gpu(0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))
        
        return np.mean(prof_res), np.std(prof_res)

if __name__ == "__main__":
    import json
    import sys

    with open(config['model_names_file'], 'r') as f:
        model_names = json.load(f)['model_names']

    stdout = sys.stdout
    f1 = open(config['exec_time_info'], 'w')

    performance = {}

    for model_name in model_names:
        print("model name : " + model_name)
        f1.write("model name : " + model_name + '\n')
        
        log_file = "%s.log" % model_name

        tuning_option = {
            'log_filename': log_file,

            'tuner': 'xgb',
            'n_trial': 200,
            'early_stopping': 60,

            'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            #runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
            runner=autotvm.RPCRunner(
                'V100_0',  # change the device key to your key
                '0.0.0.0', 9190,
                number=20, repeat=3, timeout=4, min_repeat_ms=150)
            ),
        }

        sys.stdout = f1
        performance[model_name] = list(tune_and_evaluate(tuning_option, model_name))
        sys.stdout = stdout

    sys.stdout = stdout
    
    for model, perf in performance.items():
        print(model, "%.2f (%.2f)"%(perf[0], perf[1]))
        f1.write(model + " %.2f (%.2f)"%(perf[0], perf[1]))
    