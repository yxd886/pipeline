def model_fn():
    slim = tf.contrib.slim
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    net = slim.conv2d(x, 32, [5, 5])
    net = slim.max_pool2d(net, [2, 2], 2)
    net = slim.conv2d(net, 64, [5, 5])
    net = slim.max_pool2d(net, [2, 2], 2)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.sigmoid)
    net = slim.fully_connected(net, 10, activation_fn=None)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=net,name="final_loss")
    acc = tf.reduce_mean(tf.nn.softmax(net) * y)
    optimizer = tf.train.GradientDescentOptimizer(0.002).minimize(tf.reduce_sum(loss))
    return optimizer



def get_tensors(graph,name):
    ret = []
    for op in graph.get_operations():
        for tensor in op.outputs:
            if name in tensor.name and "gradient" not in tensor.name:
                ret.append(tensor)
    return ret

import numpy as np
import tensorflow as tf
import google.protobuf.text_format as pbtf
from tensorflow.python.client import timeline
from tensorflow.distribute.cluster_resolver import TFConfigClusterResolver
from utils import info

import os


BATCHSIZE=40

devices = (
    "/job:worker/replica:0/task:0/device:GPU:0",
    "/job:worker/replica:0/task:0/device:GPU:1",
    "/job:worker/replica:0/task:0/device:GPU:2",
    "/job:worker/replica:0/task:0/device:GPU:3"
)

opt = model_fn()
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)

import tge

strategy = { node.name: [1, 1, 1, 1, 1] for node in gdef.node }

g = (tge.TGE(gdef, devices)
    .custom(strategy)
    # .replace_placeholder(BATCHSIZE)
    .use_collective()
    # .verbose()
    .compile()
    .get_result()
)


tf.reset_default_graph()
tf.import_graph_def(g)
graph = tf.get_default_graph()

x_tensor = graph.get_tensor_by_name("import/Placeholder/replica_0:0")
y_tensor = graph.get_tensor_by_name("import/Placeholder_1/replica_0:0")
opt = graph.get_operation_by_name("import/GradientDescent/replica_0")
loss = tf.reduce_mean(tf.add_n(get_tensors("final_loss")))
init = graph.get_operation_by_name("import/init/replica_0")
acc_tensor = 10 * (
    graph.get_tensor_by_name("import/Mean/replica_0:0") +
    graph.get_tensor_by_name("import/Mean/replica_1:0") +
    graph.get_tensor_by_name("import/Mean/replica_2:0") +
    graph.get_tensor_by_name("import/Mean/replica_3:0")) / 4

sess = tf.Session()
sess.run(init)

def onehot(x):
    max = x.max() + 1
    return np.eye(max)[x]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255
y_train = onehot(y_train.reshape(-1))
y_test = onehot(y_test.reshape(-1))

for batch_id in range(500000000):
    i = batch_id % 1250

    _, loss = sess.run([opt,loss], {
        x_tensor: x_train[BATCHSIZE*i:BATCHSIZE*(i+1)],
        y_tensor: y_train[BATCHSIZE*i:BATCHSIZE*(i+1)]
    })
    if batch_id%10==0:
        print("Step:{},Loss:{}".format(batch_id,loss))

