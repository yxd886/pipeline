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
sys.path.append('./bert/')

import multiprocessing as mp

config_dict =dict()
if os.path.exists("config.json"):
    with open("config.json", "r") as f:
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

devices = config_dict.get("devices", [""])

def model_fn(scope,batch_size):
    from bert.runsquad import new_model_fn_builder
    import modeling
    bert_config = modeling.BertConfig.from_json_file("bert/bert_large/bert_config.json")
    model = new_model_fn_builder(bert_config)
    features = {}
    with tf.variable_scope(scope):
        features["input_ids"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size, 128)), tf.int32)
        features["input_mask"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size, 128)), tf.int32)
        features["segment_ids"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size, 128)), tf.int32)
        features["start_positions"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size,)), tf.int32)
        features["end_positions"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size,)), tf.int32)
        loss,layer_outputs, layer_scopes= model(features)
    return loss,layer_outputs, layer_scopes


class Activater():
    def __init__(self):
        self.model_fn =model_fn
        self.devices = devices


    def build_model(self,replica_num,batch_size):
        self.losses=[]
        self.vars = []
        self.avg_gradient=[]
        self.apply_grad = []
        self.instances=[]
        self.gradients = []
        class setter():
            def __init__(self,assignment):
                self.assignment = assignment
            def choose(self,op):
                scope = tf.get_variable_scope().name
                for key in self.assignment:
                    if key in scope:
                        return self.assignment[key]
                #print(self.assignment)
                print(scope,op.name)
                return ""

        def device_setter(assignment):
            _setter = setter(assignment)
            return _setter.choose
        loss,outputs,scopes =self.model_fn("Bert",batch_size)
        tf.reset_default_graph()
        assert(len(outputs)==len(scopes))
        assignment = {item:self.devices[0] for item in scopes}
        with tf.device(device_setter(assignment)):
            loss, outputs, scopes = self.model_fn("Bert", batch_size)
        operations = tf.get_default_graph().get_operations()
        for op in operations:
            print(op.name,op.device)
    def activate_unit(self,batch_size,replica_num):
        tf.reset_default_graph()
        self.build_model(replica_num, batch_size)
        '''
        resolver = TFConfigClusterResolver()
        cluster = resolver.cluster_spec()



        config = tf.ConfigProto()
        with open("dist_config.pbtxt", "r") as f:
            txt = f.read()
        pbtf.Parse(txt, config)
        setup_workers(workers, "grpc+verbs")
        server = tf.distribute.Server(cluster, job_name='worker', task_index=0, protocol="grpc+verbs",
                                           config=config)
        target = server.target


        init_op = tf.compat.v1.global_variables_initializer()


        sess = tf.Session(target, config=config)  # , config=tf.ConfigProto(allow_soft_placement=False))


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
        '''


workers = config_dict.get("workers", ["10.28.1.26:3901","10.28.1.17:3901","10.28.1.16:3901"])

clus = dict()
clus["cluster"] = {"worker": workers}
clus["task"] = {"type": "worker", "index": 0}
os.environ["TF_CONFIG"] = json.dumps(clus)


act = Activater()
act.activate_unit(replica_num=2,batch_size=12)