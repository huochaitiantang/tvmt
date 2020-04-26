#!/bin/bash

if [ $# != 1 ]; then
  echo "please input model name, like 'resnet18_v2' !"
  exit 1
fi

echo $1

echo "python3 mxnet2onnx.py --model $1"
python3 mxnet2onnx.py --model $1




