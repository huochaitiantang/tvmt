#!/bin/bash

if [ $# != 2 ]; then
  echo "please input framework and model name, like 'mxnet resnet18' !"
  exit 1
fi

echo $1
echo $2

echo "python3 get_models.py --framework $1 --model $2"
python3 get_models.py --framework $1 --model $2




