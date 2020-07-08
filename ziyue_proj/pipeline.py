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



def setup_workers(workers, protocol="grpc"):
    import urllib.request
    import time

    param = '/'.join(server.replace(':', '%3A') for server in workers)
    for task_id, server in enumerate(workers):
        if task_id == 0: continue
        url = "http://{}:3905/{}/restart/{}/{}/{}".format(server.split(':')[0], int(time.time()) + 10, protocol, task_id, param)
        assert urllib.request.urlopen(url).read() == b'ok'
    time.sleep(1)


def model_fn(batch_size,scope):
    from bert.runsquad import new_model_fn_builder
    import modeling
    bert_config = modeling.BertConfig.from_json_file("bert/bert_large/bert_config.json")
    model = new_model_fn_builder(bert_config)
    features = {}
    with tf.variable_scope(scope):
        with tf.variable_scope("input",reuse=tf.AUTO_REUSE):
            features["input_ids"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size, 64)), tf.int32)
            features["input_mask"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size, 64)), tf.int32)
            features["segment_ids"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size, 64)), tf.int32)
            features["start_positions"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size,)), tf.int32)
            features["end_positions"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size,)), tf.int32)
        loss,layer_outputs, layer_scopes= model(features)
    return loss,[features["input_ids"]]+layer_outputs, ["input"]+layer_scopes


class Activater():
    def __init__(self,micro_batch_num,batch_size):
        self.model_fn =model_fn
        self.devices = devices
        self.micro_batch_num = micro_batch_num
        self.batch_size = batch_size
        with tf.variable_scope("Bert") as self.vs:
            pass
        loss, outputs, scopes = self.model_fn(None,self.vs)
        tf.reset_default_graph()
        self.scopes = scopes
        self.outputs = outputs

    def compute_scope_operation_dict(self):
        result = {item:[] for item in self.scopes}
        operations = self.graph.get_operations()
        for operation in operations:
            for scope in self.scopes:
                if scope in operation.name:
                    result[scope].append(operation.name)
        return result

    def compute_operation_scope_dict(self):
        result = {}
        operations = self.graph.get_operations()
        for operation in operations:
            for scope in self.scopes:
                if scope in operation.name:
                    result[operation.name] = scope
        return result
    def build_model(self):
        self.losses=[]
        self.vars = []
        self.avg_gradient=[]
        self.apply_grad = []
        self.instances=[]
        self.gradients = []
        class setter():
            def __init__(self,assignment,devices):
                self.assignment = assignment
                self.last_device =devices[0]
            def choose(self,op):
                scope = tf.get_variable_scope().name
                for key in self.assignment:
                    if key in scope:
                        self.last_device=self.assignment[key]
                        return self.assignment[key]
                #print(self.assignment)
                print(scope,op.name,self.last_device)
                return self.last_device

        def device_setter(assignment,devices):
            _setter = setter(assignment,devices)
            return _setter.choose

        assignment = {item:self.devices[0]  if i<20 else self.devices[1] for i,item in enumerate(self.scopes)}
        losses = []
        outputs = []
        for i in range(self.micro_batch_num):
            with tf.device(device_setter(assignment,self.devices)):
                loss, output, scopes = self.model_fn(None,self.vs)
                losses.append(loss)
                outputs.append(output[-1])
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.9,beta2=0.98, epsilon=1e-9).minimize(tf.add_n(losses))

        init = tf.global_variables_initializer()
        self.graph = tf.get_default_graph()
        self.gdef = tf.get_default_graph().as_graph_def(add_shapes=True)
    def change_model(self):
        strategy = {}
        assignment = {self.scopes[i]:[0,1] for i in range(len(self.scopes))}
        op_scope_dict = self.compute_operation_scope_dict()
        for op in op_scope_dict:
            place = [0]*len(self.devices)
            decision = assignment[op_scope_dict[op]]
            for i in range(decision[0],decision[1]+1,1):
                place[i] = 1
            strategy[op] = [1]+place
        for op in self.graph.get_operations():
            if op.name not in strategy.keys():
                print(op.name)
                strategy[op.name] = [1]+place

        import tge

        # options = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 1]]
        # strategy = { node.name: [np.random.randint(0, 2)] + options[np.random.randint(0, len(options))] for node in gdef.node }

        g = (tge.TGE(self.gdef, self.devices, ["Adam"])
             .custom(strategy)
             .replace_placeholder(self.batch_size)
             .use_collective()
             .compile()
             .get_result()
             )

        with open("modified.pbtxt", "w") as fo:
            fo.write(pbtf.MessageToString(g))

    def activate_unit(self):
        tf.reset_default_graph()
        self.build_model()
        self.change_model()
if __name__ == '__main__':
    config_dict =dict()
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config_dict = json.load(f)
    devices = config_dict.get("devices", [""])

    workers = config_dict.get("workers", ["10.28.1.26:3901","10.28.1.17:3901","10.28.1.16:3901"])

    clus = dict()
    clus["cluster"] = {"worker": workers}
    clus["task"] = {"type": "worker", "index": 0}
    os.environ["TF_CONFIG"] = json.dumps(clus)


    act = Activater(micro_batch_num = 2,batch_size=4)
    act.activate_unit()