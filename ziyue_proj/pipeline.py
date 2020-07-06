import numpy as np
import tensorflow as tf
import json
import os
import time
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import step_stats_pb2
import google.protobuf.text_format as pbtf
import pickle as pkl
import sys
from tensorflow.python.client import timeline
from tensorflow.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.ops import collective_ops
sys.path.append('../')
import multiprocessing as mp
arg_prefix=sys.argv[1]

config_dict =dict()
if os.path.exists("activate_config.txt"):
    with open("activate_config.txt", "r") as f:
        config_dict = json.load(f)

def setup_workers(workers, protocol="grpc"):
    import urllib.request
    import time

    param = '/'.join(server.replace(':', '%3A') for server in workers)
    for task_id, server in enumerate(workers):
        if task_id == 0: continue
        url = "http://{}:3905/{}/restart/{}/{}/{}".format(server.split(':')[0], int(time.time()) + 10, protocol, task_id, param)
        assert urllib.request.urlopen(url).read() == b'ok'
    time.sleep(1)
activate_graphs=config_dict.get("activate_graphs", ["data/graph1/nccl_dp_graph.pbtxt","data/graph1/grpc_dp_graph.pbtxt","data/graph1/single_graph.pbtxt","data/graph1/best_graph.pbtxt"])
sinks = config_dict.get("activate_sink", ["Adam"])
devices = config_dict.get("devices", [""])

def model_fn(scope,batch_size):
    from bert.runsquad import new_model_fn_builder
    import modeling
    bert_config = modeling.BertConfig.from_json_file("bert/bert_large/bert_config.json")
    model = new_model_fn_builder(bert_config)
    features = {}
    with tf.name_scope(scope):
        features["input_ids"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size, 128)), tf.int32)
        features["input_mask"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size, 128)), tf.int32)
        features["segment_ids"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size, 128)), tf.int32)
        features["start_positions"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size,)), tf.int32)
        features["end_positions"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size,)), tf.int32)
        loss = model(features)
    return loss


class Activater():
    def __init__(self):
        self.model_fn =model_fn
        self.devices = devices


    def build_model(self,replica_num,batch_size):
        self.losses=[]
        self.vars = []
        self.avg_gradient=[]
        self.apply_grad = []
        for i in range(replica_num):
            self.avg_gradient.append([])
            with tf.device(self.devices[i]):
                scope = "replica_"+str(i)
                loss =self.model_fn(scope,batch_size)
                vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                gradients = tf.compat.v1.gradients(loss, vars,colocate_gradients_with_ops=True)
                for j,gradient in enumerate(gradients):
                    sum0 = collective_ops.all_reduce(gradient, replica_num, 0, j, 'Add', 'Id')
                    self.avg_gradient[i].append(sum0)
                self.losses.append(loss)
                self.vars.append(vars)
        for i in range(replica_num):
            with tf.device(self.devices[i]):
                self.apply_grad.append(tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.9,beta2=0.98, epsilon=1e-9).apply_gradients(zip(self.avg_gradient[i],self.vars[i])))



    def activate_unit(self,batch_size,replica_num):
        setup_workers(workers, "grpc+verbs")
        tf.reset_default_graph()
        resolver = TFConfigClusterResolver()
        cluster = resolver.cluster_spec()
        '''
        dist = tf.distribute.experimental.MultiWorkerMirroredStrategy(
            tf.distribute.experimental.CollectiveCommunication.NCCL)
        config = dist.update_config_proto(tf.ConfigProto())
        config.ClearField("device_filters")
        config.allow_soft_placement = True  # log_device_placement=True)
        config.gpu_options.allow_growth = True
        '''
        config = tf.ConfigProto()
        with open("dist_config.pbtxt", "r") as f:
            txt = f.read()
        pbtf.Parse(txt, config)
        server = tf.distribute.Server(cluster, job_name='worker', task_index=0, protocol="grpc+verbs",
                                           config=config)
        target = server.target
        sess = tf.Session(target, config=config)  # , config=tf.ConfigProto(allow_soft_placement=False))

        self.build_model(replica_num,batch_size)
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        input_dict = None
        graph = tf.get_default_graph()
        placeholders = [node.outputs[0] for node in graph.get_operations() if node.node_def.op == 'Placeholder']
        shapes = [(p.shape.as_list()) for p in placeholders ]
        for shape in shapes:
            shape[0]=batch_size
        input_dict = { p: np.random.rand(*shapes[i]) for i,p in enumerate(placeholders) }

        for j in range(10):  # warm up
            sess.run(self.apply_grad, feed_dict=input_dict)

        times= []
        for j in range(10):
            tmp = time.time()
            sess.run(self.apply_grad, feed_dict=input_dict)
            times.append(time.time()-tmp)
        avg_time = sum(times)/len(times)
        print(times,"average time:", avg_time)
        print(" ")


workers = config_dict.get("workers", ["10.28.1.26:3901","10.28.1.17:3901","10.28.1.16:3901"])

clus = dict()
clus["cluster"] = {"worker": workers}
clus["task"] = {"type": "worker", "index": 0}
os.environ["TF_CONFIG"] = json.dumps(clus)


act = Activater()
act.activate_unit(replica_num=2,batch_size=12)