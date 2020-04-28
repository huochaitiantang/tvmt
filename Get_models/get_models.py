import os
import sys
import argparse
import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model


framework =['mxnet', 'onnx', 'tensorflow', 'pytorch'] 

parser = argparse.ArgumentParser()
parser.add_argument('--framework', type=str, default=None, help='a chosen framework, like mxnet, onnx or tensorflow', required=False, choices=framework)
parser.add_argument('--model', type=str, default=None, help='a chosen model, like resnet18_v2', required=False)

args = parser.parse_args()
print(args)


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
    mx.model.save_checkpoint(path+model_name, 0, mx_sym, args, auxs)
    # there are 'xx.params' and 'xx-symbol.json' on disk


def get_models_mxnet(model_name):
    block = get_model(model_name, pretrained=True)
    this_file_path = os.path.dirname(__file__)
    path_sym_params = os.path.join(this_file_path, './models/mxnet/')
    print(path_sym_params)
    save_models(block, model_name, path_sym_params)


def get_models_pytorch(model_name):
    import torch
    import torchvision.models as models

    model = getattr(models, model_name)(pretrained=True).eval()
    if "inception" in model_name:
        dummy_input = torch.randn(1, 3, 299, 299)
    else:
        dummy_input = torch.randn(1, 3, 224, 224)
    
    script = torch.jit.trace(model, dummy_input)
    current_path = os.path.dirname(__file__)
    pytorch_path = os.path.join(current_path, './models/pytorch/')
    onnx_path = os.path.join(current_path, './models/onnx/')
    
    if not os.path.exists(pytorch_path):
        os.makedirs(pytorch_path)
    if not os.path.exists(onnx_path):
        os.makedirs(onnx_path)

    script.save(pytorch_path + model_name + '.pt')

    torch.onnx.export(script, dummy_input, onnx_path + model_name + '.onnx', verbose=True, input_names=['data'], output_names=['output1'], example_outputs=script(dummy_input))
    

'''
def untar(fname, dirs):
    import tarfile
    import shutil
    try:
        t = tarfile.open(fname)
        names = t.getnames()
        for name in names:
            t.extract(name,path=dirs)
        t.close()
        for name in names:
            os.popen('mv '+os.path.join(dirs,name)+' '+dirs)
        #shutil.rmtree(os.path.join(dirs,names[0]))
        return True
    except Exception as e:
        print(e)
        return False

#These models are provided by coco
def get_models_tensorflow(model_name):
    #get_model_list
    from urllib.request import urlretrieve
    list_path='./models/tensorflow/models_name'
    tf_path='./models/tensorflow/'
    models = []
    with open(list_path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            model_link=line.split(' ')
            if model_name in model_link:
                filepath = os.path.join(tf_path, model_name+'.tar.gz')
                if not os.path.exists(tf_path):
                    os.makedirs(tf_path)
                if not os.path.exists(filepath):    
                    print(model_link[1])
                    urlretrieve(model_link[1],filepath)
                else:
                    print(model_name+" exists!")
                if untar(filepath,os.path.join(tf_path, model_name+'/')):
                    #os.remove(filepath)
                    print("model ready.")
                return
            #models.append(model_link)
    print("model not included!")
    sys.exit()
'''

#These models are provided by tvm

def get_models_tensorflow(model_name):
    #from tvm.contrib.download import download_testdata
    from urllib.request import urlretrieve
    repo_base = 'https://github.com/dmlc/web-data/raw/master/tensorflow/models'
    list_path='./models/tensorflow/models_name'
    tf_path='./models/tensorflow/'
    models = []
    with open(list_path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            model_link=line.split(' ')
            if model_name in model_link:
                 if not os.path.exists(tf_path):
                     os.makedirs(tf_path)
                 filepath = os.path.join(tf_path,model_name+'.pb')
                 model_url = repo_base+model_link[1]
                 print(model_url)
                 print(filepath)
                 urlretrieve(model_url,filepath)
                 print('file in '+ filepath)
                 return
        print('model not included')
        sys.exit()


def main():
    if args.framework == 'mxnet':
        get_models_mxnet(args.model)
    elif args.framework == 'tensorflow':
        get_models_tensorflow(args.model)
    elif args.framework == 'pytorch':
        get_models_pytorch(args.model)


if __name__ == '__main__':
    main()
