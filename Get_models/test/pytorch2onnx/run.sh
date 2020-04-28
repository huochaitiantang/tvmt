#!/bin/bash

if [ $# == 0 ]; then
  echo "please input model name, like 'resnet18_v2' !"
  exit 1
fi

echo $1
if [ $# == 1 ]; then
    echo "python3 pytorch2onnx.py --model $1"
    python3 pytorch2onnx.py --model $1
elif [ $# == 2 ]; then
    echo $2
    echo "python3 pytorch2onnx.py --model $1 --verbose $2"
    python3 pytorch2onnx.py --model $1 --verbose $2
fi

