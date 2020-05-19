import os
import sys


def mv(path, thread):
    dir = os.listdir(path)
    print(dir)

    for line in dir:
        model_name = line[ : line.find('.log') ]
        print(model_name)
        model_name = model_name[ model_name.find('.')+1 : ]
        print(model_name)
        cmd = 'mv ' + path +line + ' ' + path+'aarch64_onnx_1batch_'+thread+'_'+model_name+'.log'
        print(cmd)
        os.system(cmd)


path = './1thread/'
thread = '1thread'
mv(path, thread)
path = './4thread/'
thread = '4thread'
mv(path, thread)
