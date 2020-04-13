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
import sys
from tvm import rpc, autotvm
from topi.util import get_const_tuple
from tvm.contrib import util

######################################################################
# Download Resnet18 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
from tvm.contrib.download import download_testdata
from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
from matplotlib import pyplot as plt

img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_name = 'cat.png'
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'imagenet1000_clsid_to_human.txt'
img_path = download_testdata(img_url, 'cat.png', module='data')
synset_path = download_testdata(synset_url, synset_name, module='data')
with open(synset_path) as f:
    synset = eval(f.read())


def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


def trans_image(model_name):
    img_size = 299 if model_name == 'inceptionv3' else 224
    image = Image.open(img_path).resize((img_size, img_size))
    #plt.imshow(image)
    #plt.show()
    x = transform_image(image)
    print('x', x.shape)
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
    #target = 'cuda'
    #target = 'llvm'
    target = tvm.target.arm_cpu("rk3399")
    #target = tvm.target.arm_cpu("rasp")

    #original 
    host = os.environ['PI']
    port = 9090


    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(func, target, params=params)

    '''
    # Save the library at local temporary directory.
    tmp = util.tempdir()
    lib_fname = tmp.relpath('net.tar')
    lib.export_library(lib_fname)

    ######################################################################
    # Execute the portable graph on TVM
    # ---------------------------------
    # Now, we would like to reproduce the same forward computation using TVM.
    from tvm.contrib import graph_runtime
    #from tvm.contrib.debugger import debug_runtime
    #ctx = tvm.gpu(0)
    #ctx = tvm.cpu(0)
    remote = rpc.connect(host, port)

    # upload the library to remote device and load it
    remote.upload(lib_fname)
    rlib = remote.load_module('net.tar')

    ctx = remote.cpu(0)
    dtype = 'float32'
    m = graph_runtime.create(graph, rlib, ctx)
    #m = debug_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input(**params)
    m.set_input('data', tvm.nd.array(x.astype(dtype)))
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0)
    top1 = np.argmax(tvm_output.asnumpy()[0])
    print('TVM prediction top-1:', top1, synset[top1])

    print('Evaluate inference time cost...')
    ftimer = m.module.time_evaluator('run', ctx, number=2, repeat=10)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print('Mean inference time (std dev): %.2f ms (%.2f ms)' % (np.mean(prof_res),
                                                            np.std(prof_res)))
    '''

    # Save the library at local temporary directory.
    tmp = util.tempdir()
    lib_fname = tmp.relpath('net.tar')
    lib.export_library(lib_fname)
    
    ######################################################################
    # Deploy the Model Remotely by RPC
    # --------------------------------
    # With RPC, you can deploy the model remotely from your host machine
    # to the remote device.
    
    # obtain an RPC session from remote device.
    remote = rpc.connect(host, port)
    
    # upload the library to remote device and load it
    remote.upload(lib_fname)
    rlib = remote.load_module('net.tar')
    
    # create the remote runtime module
    ctx = remote.cpu(0)
    from tvm.contrib import graph_runtime
    print("before graph_runtime")
    module = graph_runtime.create(graph, rlib, ctx)
    print("after graph_runtime")
    # set parameter (upload params to the remote device. This may take a while)
    module.set_input(**params)
    # set input data
    module.set_input('data', tvm.nd.array(x.astype('float32')))
    # run
    module.run()
    # get output
    out = module.get_output(0)
    # get top1 result
    top1 = np.argmax(out.asnumpy())
    print('TVM prediction top-1: {}'.format(synset[top1]))
    


def main():
    '''
    model_names = [
        'squeezenet1.0',
        'resnet18_v1',
        'resnet50_v2',
        'mobilenetv2_1.0',
        'mobilenet1.0',
        'inceptionv3',
        'densenet121'
        
    ]
    '''

    model_names = [
        'inceptionv3',
        'mobilenet0.25',
        'mobilenet0.5',
        'mobilenet0.75',
        'mobilenet1.0',
        'mobilenetv2_0.25',
        'mobilenetv2_0.5',
        'mobilenetv2_0.75',
        'mobilenetv2_1.0',
        'resnet101_v1',
        'resnet101_v2',
        'resnet152_v1',
        'resnet152_v2',
        'resnet18_v1',
        'resnet18_v2',
        'resnet34_v1',
        'resnet34_v2',
        'resnet50_v1',
        'resnet50_v2',
        'squeezenet1.0',
        'squeezenet1.1'
        ]

    model_names = [
        'resnet18_v1'
        ]


    for model_name in model_names:
        print("model name : "+model_name)
        x = trans_image(model_name)
        block = get_model(model_name, pretrained=True)
        test_relay(block, x)


if __name__ == "__main__":
    main()

