import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
args = parser.parse_args()
print(args)


def get_model_path(model_name):
    return '../models/onnx/' + model_name + '.onnx'


def get_engine(model_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                print("parser model failed")

        engine = builder.build_cuda_engine(network)
    
    return engine

def allocate_buffer(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    return h_input, h_output, d_input, d_output, stream


def inference(engine, h_input, h_output, d_input, d_output, stream):
    context = engine.create_execution_context()
    t = time.time()
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()
    # Return the host output. 
    t = time.time() - t

    return t

def check_output(output, model_file_path, data):
    import onnxruntime

    session = onnxruntime.InferenceSession(model_file_path)
    inputs = {session.get_inputs()[0].name: data}
    result = np.array(session.run([], inputs))

    mse = np.mean((output - result) ** 2)
    precision_meet = np.allclose(output, result, atol=1e-05)
    return mse, precision_meet

if __name__ == '__main__':
    print('--------------------------------------------------')
    if not args.model:
        print('We must have a model')
        exit()

    model_file_path = get_model_path(args.model)
    print("build engine")
    engine = get_engine(model_file_path)
    
    print('allocate buffer')
    h_input, h_output, d_input, d_output, stream = allocate_buffer(engine)
    
    if "inception" in args.model:
        input_shape = (1, 3, 299, 299)
    else:
        input_shape = (1, 3, 224, 224)
    data = np.random.uniform(size=input_shape).astype('float32')
    np.copyto(h_input, data.reshape(-1))

    print('do inference')
    t = inference(engine, h_input, h_output, d_input, d_output, stream)

    print('check output')
    mse, precision_assert = check_output(h_output, model_file_path, data)
    
    if precision_assert:
        print('pass the output check, the mse is {:.2e}'.format(mse))
    else:
        print('failed the ouput check, the mse is {:.2e}'.format(mse))

    print('TenserRT inference time is {:.2f}ms'.format(t * 1000))
    print('--------------------------------------------------')

