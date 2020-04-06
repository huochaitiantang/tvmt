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
import os

######################################################################
# Download Resnet18 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
from tvm.contrib.download import download_testdata
from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet

######################################################################
# Use MXNet symbol with pretrained weights
# ----------------------------------------
# MXNet often use `arg_params` and `aux_params` to store network parameters
# separately, here we show how to use these weights with existing API


def block2symbol(block):
    data = mx.sym.Variable('data')
    sym = block(data)
    args = {}
    auxs = {}
    for k, v in block.collect_params().items():
        args[k] = mx.nd.array(v.data().asnumpy())
    return sym, args, auxs


def save_models(block, model_name, path):
    mx_sym, args, auxs = block2symbol(block)
    # usually we would save/load it as checkpoint
    os.makedirs(path, exist_ok=True)
    mx.model.save_checkpoint(path+"/"+model_name, 0, mx_sym, args, auxs)
    # there are 'xx.params' and 'xx-symbol.json' on disk


def convert_sym_params_to_onnx(model_name, path_sym_params, path_onnx):
    img_size = 299 if model_name == 'inceptionv3' else 224
    input_shape = (1, 3, img_size, img_size)
    # symbol and params
    sym = path_sym_params + '/' + model_name + '-symbol.json'
    params = path_sym_params + '/' + model_name + '-0000.params'
    # Path of the output file
    onnx_file = path_onnx+'/'+model_name+'.onnx'
    # Invoke export model API. It returns path of the converted onnx model
    converted_model_path = onnx_mxnet.export_model(
        sym, params, [input_shape], np.float32, onnx_file, verbose=True)


def main():
    model_names = [
        'resnet18_v1',
        'resnet50_v2',
        'mobilenetv2_1.0',
        'mobilenet1.0',
        'inceptionv3',
        'densenet121'
    ]
    this_file_path = os.path.dirname(__file__)
    tvmt_path = os.path.join(this_file_path, os.path.pardir)
    path_onnx = os.path.join(tvmt_path, 'onnx')
    os.makedirs(path_onnx, exist_ok=True)

    for model_name in model_names:
        print("model name : "+model_name)
        block = get_model(model_name, pretrained=True)
        path_sym_params = os.path.join(this_file_path, './symbol_and_params')
        save_models(block, model_name, path_sym_params)
        convert_sym_params_to_onnx(model_name, path_sym_params, path_onnx)


if __name__ == "__main__":
    main()


######################################################################
# for a normal mxnet model, we start from here
#mx_sym, args, auxs = mx.model.load_checkpoint('resnet18_v1', 0)
# now we use the same API to get Relay computation graph
#mod, relay_params = relay.frontend.from_mxnet(mx_sym, shape_dict,
#                                              arg_params=args, aux_params=auxs)
# repeat the same steps to run this model using TVM
