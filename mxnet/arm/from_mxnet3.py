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

#from tvm import te
from tvm import autotvm
#from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime


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


def test_relay(model_name, x):
    #vgg
    if "vgg" in model_name:
        n_layer = int(model_name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=1, dtype='float32')
        print(n_layer)
    else:
        block = get_model(model_name, pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape_dict)


    ######################################################################
    # Compile the Graph
    # -----------------
    # Now we would like to port the Gluon model to a portable computational graph.
    # It's as easy as several lines.
    # We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
    shape_dict = {'data': x.shape}
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


    '''
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(func, target, params=params)

    # Save the library at local temporary directory.
    tmp = util.tempdir()
    lib_fname = tmp.relpath('net.tar')
    lib.export_library(lib_fname)
    
    ######################################################################
    # Deploy the Model Remotely by RPC
    # --------------------------------
    # With RPC, you can deploy the model remotely from your host machine
    # to the remote device.

    key = 'rk3399'
    print("before graph_runtime")
    #tracker = rpc.connect_tracker(host, port)
    print("after graph_runtime")
    # When running a heavy model, we should increase the `session_timeout`
    #remote = tracker.request(key, priority=0,
    #                         session_timeout=60)

   
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
    '''
    



    # compile kernels with history best records
    print("Compile...")
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build_module.build(
            mod, target=target, params=params)

    # export library
    #tmp = tempdir()
    tmp = util.tempdir()
    use_android = False
    if use_android:
        from tvm.contrib import ndk
        filename = "net.so"
        lib.export_library(tmp.relpath(filename), ndk.create_shared)
    else:
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

    # upload module to device
    print("Upload...")
    device_key = 'rk3399'
    remote = autotvm.measure.request_remote(device_key, '0.0.0.0', 9190,
                                            timeout=1000000)
    remote.upload(tmp.relpath(filename))
    rlib = remote.load_module(filename)

    # upload parameters to device
    ctx = remote.context(str(target), 0)
    module = runtime.create(graph, rlib, ctx)

    #
    #data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    #module.set_input('data', data_tvm)
    #
    module.set_input('data', tvm.nd.array(x.astype('float32')))

    module.set_input(**params)

    # run
    module.run()
    # get output
    out = module.get_output(0)
    # get top1 result
    top1 = np.argmax(out.asnumpy())
    print('TVM prediction top-1: {}'.format(synset[top1]))

    # evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
          (np.mean(prof_res), np.std(prof_res)))


def main():

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
        'squeezenet1.1',
        'densenet121',
        'densenet161',
        'densenet169',
        'densenet201',
        ]

    model_names=[

        #'vgg11',
        #'vgg11_bn',
        #'vgg13',
        #'vgg13_bn',
        #'vgg16',
        #'vgg16_bn',
        #'vgg19',
        #'vgg19_bn',
        'alexnet',
        #'vgg-16',
            ]


    for model_name in model_names:
        print("model_name : "+model_name)
        x = trans_image(model_name)
        test_relay(model_name, x)

if __name__ == "__main__":
    main()

