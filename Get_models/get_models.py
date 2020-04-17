import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--framework', type=str, default=None, help='a chosen framework, like mxnet, onnx or tensorflow', required=False)
parser.add_argument('--model', type=str, default=None, help='a chosen model, like resnet18', required=False)

args = parser.parse_args()
print(args)

print( args.framework)
print( args.model)
framework =['mxnet', 'onnx', 'tensorflow'] 
models = ['resnet18']
if args.framework not in framework:
    print( str(args.framework) + " not in " + str(framework) )
    sys.exit()
if args.model not in models:
    print( str(args.model) + " not in " + str(models) )
    sys.exit()

