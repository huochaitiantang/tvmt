import os
from tensorflow.python.tools import freeze_graph



freeze_graph.freeze_graph("./models/tensorflow/nf_model/inceptionv2",
                          "",
                          "true",
                          "./models/tensorflow/inception_v2.ckpt",
                          "InceptionV2/Predictions/Reshape_1",
                          "save/restore_all",
                          "",
                          "./models/tensorflow/inception_v2.pb",
                          "",
                          "",
                          "",
                          "",
                          "")


