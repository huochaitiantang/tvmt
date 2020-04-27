from tf2onnx.tfonnx import process_tf_graph,tf_optimize
from tf2onnx import constants, logging, utils, optimizer
from tf2onnx import loader as tf_loader
import tensorflow as tf
import argparse
from tensorflow.graph_util import convert_variables_to_constants as freeze_graph

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None, help='input a model, default=inceptionv2', required=False)
parser.add_argument('--model_path', type=str, default=None, help='input a model, default=../models/tensorflow/inceptionv2/frozen_inference_graph.pb', required=False)
parser.add_argument('--output', type=str,default=None, help='input an output path, default=./inceptionv2.onnx', required=False)
parser.add_argument('--inputs', type=str,default='image_tensor:0', help='input an tensor name, default=image_tensor:0', required=False)
parser.add_argument('--outputs', type=str,default='detection_boxes:0,detection_classes:0,detection_scores:0,num_detections:0', help='input an output path, default=detection_boxes:0,detection_classes:0,detection_scores:0,num_detections:0', required=False)
args = parser.parse_args()

if args.inputs:
    args.inputs, args.shape_override = utils.split_nodename_and_shape(args.inputs)
if args.outputs:
    args.outputs = args.outputs.split(",")
if args.model:
    model_path='../models/tensorflow/'+args.model+'/frozen_inference_graph.pb'
    graph_def, inputs, outputs = tf_loader.from_graphdef(model_path, args.inputs, args.outputs)
if args.model_path:
    graph_def, inputs, outputs = tf_loader.from_graphdef(args.model_path, args.inputs, args.outputs)



graph_def = tf_optimize(inputs, outputs, graph_def, "")

with tf.Graph().as_default() as tf_graph:
    tf.import_graph_def(graph_def, name='')
with tf.Session(graph=tf_graph):
    g = process_tf_graph(tf_graph, opset=10, input_names=inputs, output_names=outputs)
    onnx_graph = optimizer.optimize_graph(g)
    model_proto = onnx_graph.make_model("converted from {}".format(args.model_path))
    utils.save_protobuf(args.output, model_proto)
    print("Converted Model saved in %s" % args.output)

