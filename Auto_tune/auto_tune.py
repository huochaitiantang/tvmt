import os
import sys
import argparse
import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model
from tvm.contrib import util
import tvm.relay as relay
import tvm
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default=None, help='a chosen target, like x86, gpu, arm or aarch64', required=False)
parser.add_argument('--framework', type=str, default=None, help='a chosen framework, like mxnet, onnx or tensorflow', required=False)
parser.add_argument('--model', type=str, default=None, help='a chosen model, like resnet18_v2', required=False)

args = parser.parse_args()
print(args)


def getData(path, data_lists):
    with open(path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            data_lists.append(line)

def get_model_names():
    path = '../Get_models/models/mxnet/model_names'
    model_names = []
    getData(path, model_names)
    return model_names

models = get_model_names()
print(models)

print( args.target)
print( args.framework)
print( args.model)
framework =['mxnet', 'onnx', 'tensorflow'] 
target = ['x86', 'gpu', 'arm', 'aarch64']

if args.target not in target:
    print( str(args.target) + " not in " + str(target) )
    sys.exit()
if args.framework not in framework:
    print( str(args.framework) + " not in " + str(framework) )
    sys.exit()
if args.model not in models:
    print( str(args.model) + " not in " + str(models) )
    sys.exit()


def get_models_mxnet(model_name, shape_dict):
    path = '../Get_models/models/mxnet/'
    # for a normal mxnet model, we start from here
    #mx_sym, args, auxs = mx.model.load_checkpoint('resnet18_v1', 0)
    mx_sym, args, auxs = mx.model.load_checkpoint( path + model_name , 0 )
    # now we use the same API to get Relay computation graph
    mod, relay_params = relay.frontend.from_mxnet(mx_sym, shape_dict,
                                              arg_params=args, aux_params=auxs)
    return mod , relay_params

def get_target():
    # now compile the graph
    if args.target == 'x86':
        target = tvm.target.create('llvm')
    elif args.target == 'gpu':
        target = tvm.target.cuda()
    elif args.target == 'arm':
        target = tvm.target.create('llvm -device=arm_cpu -target=armv7l-linux-gnueabihf -mattr=+neon')
    elif args.target == 'aarch64':
        target = tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu -mattr=+neon')
    return target

def relay_save_lib(model_name, mod, params, log_file = None):
    ## we want a probability so add a softmax operator
    func = mod["main"]
    func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)

    target = get_target()

    if log_file is not None :
        with autotvm.apply_history_best(log_file):
            print("Compile with log_file...")
            with relay.build_config(opt_level=3):
                graph, lib, params = relay.build(func, target, params=params)
    else:
        print("Compile without log_file...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(func, target, params=params)

    # save the graph, lib and params into separate files
    deploy_name = args.target + '_' + args.framework + '_' + model_name
    path = './lib_json_params/' + args.target + '/' + args.framework + '/'


    if args.target == 'arm' or args.target == 'aarch64':
        lib.export_library( path + deploy_name + '.tar' )
    elif args.target == 'x86':
        lib.export_library( path + deploy_name + '.tar' )
        #lib.export_library( path + deploy_name + '.dylib' )
        #lib.save( path + deploy_name + '.ll' )
    elif args.target == 'gpu':
        lib.export_library( path + deploy_name + '.tar' )

    with open( path + deploy_name + ".json", "w") as fo:
        fo.write(graph)
    with open( path + deploy_name + ".params", "wb") as fo:
        fo.write(relay.save_param_dict(params))


def get_network(model_name, batch_size, input_name):
    input_shape = ( batch_size, 3, 224, 224 )
    if 'inception' in model_name:
        input_shape = ( batch_size, 3, 299, 299 )

    shape_dict = {input_name: input_shape}
    mod, params = get_models_mxnet(model_name, shape_dict)
    return mod, params, input_shape

def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=True,
               try_spatial_pack_depthwise=False):
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass

    # if we want to use spatial pack for depthwise convolution
    if try_spatial_pack_depthwise:
        tuner = 'xgb_knob'
        for i in range(len(tasks)):
            if tasks[i].name == 'topi_nn_depthwise_conv2d_nchw':
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host,
                                          'contrib_spatial_pack')
                tasks[i] = tsk

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        n_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def tuning( tuning_option, 
            model_name = None,
            batch_size =1 ,
            dtype = 'float32',
            input_name = 'date',
            device_key = None,
            use_android = False ):
        
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape  = get_network(model_name, batch_size, input_name)
    target = get_target()
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.nn.conv2d,))

    # get log_filename
    log_file = tuning_option['log_filename']

    # run tuning tasks
    if os.path.exists(log_file):
        print(log_file + " exists, skipping...")
    else:
        print("Tuning...")
        tune_tasks(tasks, **tuning_option)

    relay_save_lib(model_name, mod, params, log_file )


def get_log_file(model_name):
    print("model_name : "+model_name)
    log_file_path = './log/' + args.target + '/' + args.framework + '/'
    log_file = log_file_path + args.target + '_' + args.framework + '_' + model_name +".log"
    print(log_file)
    return log_file


def tuning_mxnet(model_name):
    batch_size = 1
    dtype = "float32"
    # Set the input name of the graph
    # For ONNX models, it is typically "input1".
    input_name = "data"
    device_key= None
    if args.target == 'arm':
        device_key = 'rasp3b'
    elif args.target == 'aarch64':
        device_key = 'rk3399'
    elif args.target == 'gpu':
        device_key = '1080ti'
    use_android = False

    log_file = get_log_file(model_name)

    other_option = {
        'model_name' : model_name,
        'batch_size' : batch_size,
        'dtype' : dtype,
        'input_name' : input_name,
        'device_key' : device_key,
        'use_android' : use_android
    }

    if args.target == 'arm' or args.target == 'aarch64':
        measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(
                    build_func='ndk' if use_android else 'default'),
                runner=autotvm.RPCRunner(
                    device_key, host='0.0.0.0', port=9190,
                    number=5,
                    timeout=10,
                ),
            )

    elif args.target == 'x86':
        measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.LocalRunner(number=10, repeat=1,
                                           min_repeat_ms=1000),
            )

    elif args.target == 'gpu':
        measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                #runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
                runner=autotvm.RPCRunner(
                    '1080ti',  # change the device key to your key
                    '0.0.0.0', 9190,
                    number=20, repeat=3, timeout=4, min_repeat_ms=150)
            )


    tuning_option = {
        'log_filename': log_file,
        'tuner': 'xgb',
        'n_trial': 1,
        'early_stopping': 150,
        'measure_option': measure_option
    }

    tuning( tuning_option, **other_option )



def main():
    model_name = args.model
    if args.framework == 'mxnet':
        tuning_mxnet(model_name)
    elif args.framework == 'tensorflow':
        tuning_tensorflow(model_name)
    elif args.framework == 'onnx':
        tuning_onnx(model_name)

if __name__ == '__main__':
    main()

