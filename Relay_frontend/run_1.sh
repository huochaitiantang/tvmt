#!/bin/bash

if [ $# != 3 ]; then
  echo "please input target, framework and model name, like 'x86 mxnet resnet18_v2' !"
  exit 1
fi

echo $1
echo $2
echo $3

echo "python3 relay_frontend.py --target $1 --framework $2 --model $3"
python3 relay_frontend_1.py --target $1 --framework $2 --model $3




