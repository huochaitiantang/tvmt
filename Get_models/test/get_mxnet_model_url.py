import os
import sys
import argparse
import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model




def getData(path, data_lists):
    with open(path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            data_lists.append(line)

def get_model_names():
    path = '../models/mxnet/model_names'
    model_names = []
    getData(path, model_names)
    return model_names

models = get_model_names()


def get_models_mxnet(model_name):
    block = get_model(model_name, pretrained=True)

def get_url():
    path = './raw_data_url'
    raw_data = []
    getData(path, raw_data)
    urls = []
    for line in raw_data:
        index = line.find('http')
        tmp = line[index:]
        tmp = tmp.strip('...')
        urls.append(tmp)
    return urls

def main():
    model_names = [
        #'inceptionv3',
        #'mobilenet0.25',
        #'mobilenet0.5',
        #'mobilenet0.75',
        #'mobilenet1.0',
        #'mobilenetv2_0.25',
        #'mobilenetv2_0.5',
        #'mobilenetv2_0.75',
        #'mobilenetv2_1.0',
        #'resnet101_v1',
        #'resnet101_v2',
        #'resnet152_v1',
        #'resnet152_v2',
        #'resnet18_v1',
        #'resnet18_v2',
        #'resnet34_v1',
        #'resnet34_v2',
        #'resnet50_v1',
        #'resnet50_v2',
        #'squeezenet1.0',
        #'squeezenet1.1',
        #'densenet121',
        #'densenet161',
        #'densenet169',
        #'densenet201',

        #'vgg11',
        #'vgg11_bn',
        #'vgg13',
        #'vgg13_bn',
        #'vgg16',
        #'vgg16_bn',
        #'vgg19',
        #'vgg19_bn',
        #'alexnet'
            ]


    #for model_name in model_names:
    #    get_models_mxnet(model_name)

    urls = get_url()
    for url in urls:
        print(url)

if __name__ == '__main__':
    main()

