'''import tf2onnx
import tensorflow as tf
#from tensorflow.graph_util import convert_variables_to_constants as freeze_graph


model_path='/home/test/tf/models/tensorflow/inceptionv2/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
print(model_path)
with tf.compat.v1.gfile.GFile(model_path, 'rb') as f:
	graph_def = tf.compat.v1.GraphDef()
	#print(f.read())
	graph_def.ParseFromString(f.read())
	graph = tf.import_graph_def(graph_def, name='')
	# Call the utility to import the graph definition into default graph.
	#graph_def = tf_testing.ProcessGraphDefParam(graph_def)
	# Add shapes to the graph.
	#with tf.compat.v1.Session() as sess:
	#    graph_def = tf_testing.AddShapesToGraphDef(sess, 'softmax')
	print("finished")
	onnx_graph = process_tf_graph(graph, opset=7, input_names="X:0", output_names=":0")
	model_proto = onnx_graph.make_model("test")
	print("ONNX model is saved at ./output/mnist4.onnx")
	with open("./output/mnist4.onnx", "wb") as f:
	    f.write(model_proto.SerializeToString())


import os
os.popen("python3 -m tf2onnx.convert --opset 10 --fold_const --graphdef ~/tf/models/tensorflow/inceptionv2/frozen_inference_graph.pb --output ~/save.onnx --inputs image_tensor:0 --outputs detection_boxes:0,detection_classes:0,detection_scores:0,num_detections:0")

'''
import subprocess
import argparse
import tensorflow as tf
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='inceptionv2', help='input a model, default=inceptionv2', required=False)
parser.add_argument('--output', type=str,default='./inceptionv2.onnx', help='input an output path, default=./inceptionv2.onnx', required=False)
args = parser.parse_args()


print (args.output)
comm = 'python -m tf2onnx.convert --opset 10 --fold_const --graphdef ../models/tensorflow/'+args.model+'/frozen_inference_graph.pb --output '+args.output+' --inputs image_tensor:0 --outputs detection_boxes:0,detection_classes:0,detection_scores:0,num_detections:0'
print (comm)


p = subprocess.Popen(comm, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)   

print (p.stdout.readlines())
for line in p.stdout.readlines():   
    print (line)  
retval = p.wait()


