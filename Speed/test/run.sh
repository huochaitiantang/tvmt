#!/bin/bash

if [ $# != 3 ] && [ $# != 4 ]; then
  echo "please input target, framework, model name and tuned or not, like 'x86 mxnet resnet18_v2 tuned' !"
  exit 1
fi

echo $1
echo $2
echo $3

if [ $# == 3 ]; then
echo "Not tuned"
echo "python3 speed.py --target $1 --framework $2 --model $3 --tuned No"
python3 speed.py --target $1 --framework $2 --model $3 --tuned No

elif [ $# == 4 ]; then
echo $4
echo "python3 speed.py --target $1 --framework $2 --model $3 --tuned Yes"
python3 speed.py --target $1 --framework $2 --model $3 --tuned Yes

fi

