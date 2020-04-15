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
.. _tutorial-from-mxnet:

Compile MXNet Models
====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_, \
            `Kazutaka Morita <https://github.com/kazum>`_

This article is an introductory tutorial to deploy mxnet models with Relay.

For us to begin with, mxnet module is required to be installed.

A quick solution is

.. code-block:: bash

    pip install mxnet --user

or please refer to offical installation guide.
https://mxnet.incubator.apache.org/versions/master/install/index.html
"""
# some standard imports
import mxnet as mx
import tvm
import tvm.relay as relay
import numpy as np

######################################################################
# Download Resnet18 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
from tvm.contrib.download import download_testdata
from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
from matplotlib import pyplot as plt

import argparse

config = {
    'model_names_file':'model_names.json',
    'exec_time_info':'./log/exec_info',
    'config_err':'./log/target_config_err',
    'target':'cuda',
    'ctx_dev':'gpu',
    'ctx_no':0
}

parser = argparse.ArgumentParser()
for key, value in config.items():
    if type(value) == int:
        parser.add_argument('-' + key, default=value, type=int)
    else:
        parser.add_argument('-' + key, default=value)

args = parser.parse_args()
target = args.target
if args.ctx_dev == 'cpu':
    ctx = tvm.cpu(args.ctx_no)
elif args.ctx_dev == 'gpu':
    ctx = tvm.gpu(args.ctx_no)
    
config["model_names_file"] = args.model_names_file
config["exec_time_info"] = args.exec_time_info
config['config_err'] = args.config_err



def trans_image(model_name):
    img_size = 299 if model_name == 'inceptionv3' else 224
    x = np.random.uniform(size=(1,3,img_size, img_size))
    return x


def test_relay(block, x):
    ######################################################################
    # Compile the Graph
    # -----------------
    # Now we would like to port the Gluon model to a portable computational graph.
    # It's as easy as several lines.
    # We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
    shape_dict = {'data': x.shape}
    mod, params = relay.frontend.from_mxnet(block, shape_dict)
    ## we want a probability so add a softmax operator
    func = mod["main"]
    func = relay.Function(func.params, relay.nn.softmax(
        func.body), None, func.type_params, func.attrs)

    ######################################################################
    # now compile the graph
    target = 'cuda'
    # target = 'llvm'
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(func, target, params=params)
    # graph, lib, params = relay.build(func, target, params=params)

    ######################################################################
    # Execute the portable graph on TVM
    # ---------------------------------
    # Now, we would like to reproduce the same forward computation using TVM.
    from tvm.contrib import graph_runtime
    # Debug import
    # from tvm.contrib.debugger import debug_runtime as graph_runtime
    dtype = 'float32'
    # m = graph_runtime.create(graph, lib, ctx)
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input('data', tvm.nd.array(x.astype(dtype)))
    m.set_input(**params)
    # execute
    ftimer = m.module.time_evaluator('run', ctx, number=1,repeat=600)
    prof_res = np.array(ftimer().results)
    print('Mean inference time (std dev): %.2f us (%.2f us)'%(np.mean(prof_res), np.std(prof_res)))
    return np.mean(prof_res), np.std(prof_res)

def main():
    import json
    import sys

    with open(config['model_names_file'], 'r') as f:
        model_names = json.load(f)['model_names']

    stdout = sys.stdout
    stderr = sys.stderr
    f1 = open(config['exec_time_info'], 'w')
    f2 = open(config['config_err'], 'w')

    performance = {}

    for model_name in model_names:
        print("model name : " + model_name)
        f1.write("model name : " + model_name + '\n')
        f2.write("model name : " + model_name + '\n')
        x = trans_image(model_name)
        block = get_model(model_name, pretrained=True)

        sys.stdout = f1
        sys.stderr = f2
        performance[model_name] = list(test_relay(block, x))
        sys.stdout = stdout
        sys.stderr = stderr

    sys.stdout = stdout
    sys.stderr = stderr
    
    for model, perf in performance.items():
        print(model, "%.2f (%.2f)"%(perf[0], perf[1]))
    
    f1.write(model + " %.2f (%.2f)"%(perf[0], perf[1]))


if __name__ == "__main__":
    main()
