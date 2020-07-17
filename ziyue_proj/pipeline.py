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
sys.path.append('./vgg_19/')
sys.path.append('./resnet/')

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


def model_fn(batch_size,model_name):
    if model_name=="bert":
        from bert.runsquad import new_model_fn_builder
        import modeling
        bert_config = modeling.BertConfig.from_json_file("bert/bert_large/bert_config.json")
        model = new_model_fn_builder(bert_config)
        features = {}
        if True:
            with tf.variable_scope("input",reuse=tf.AUTO_REUSE):
                features["input_ids"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size, 128)), tf.int32)
                features["input_mask"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size, 128)), tf.int32)
                features["segment_ids"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size, 128)), tf.int32)
                features["start_positions"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size,)), tf.int32)
                features["end_positions"] = tf.cast(100 * tf.placeholder(tf.float32, shape=(batch_size,)), tf.int32)
            loss,layer_outputs, layer_scopes= model(features)
            return loss, [features["input_ids"]] + layer_outputs, ["input"] + layer_scopes

    elif model_name=="vgg_19":
        import vgg
        with tf.variable_scope("input", reuse=tf.AUTO_REUSE):
            x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
            y = tf.placeholder(tf.float32, shape=(batch_size,1,1,1000))
        loss, endpoints,scopes = vgg.vgg_19(x,y, 1000)

        return loss, [x] + endpoints, ["input"] + scopes

    elif model_name=="resnet":
        import resnet_v2
        with tf.variable_scope("input", reuse=tf.AUTO_REUSE):
            x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
            y = tf.placeholder(tf.float32, shape=(batch_size,1,1,1000))
        loss, endpoints,scopes = resnet_v2.resnet_v2_152(x,y, 1000)

        return loss, [x] + endpoints, ["input"] + scopes

class Activater():
    def __init__(self,micro_batch_num,batch_size,model_name):
        self.model_fn =model_fn
        self.devices = devices
        self.micro_batch_num = micro_batch_num
        self.batch_size = batch_size
        self.model_name = model_name

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
            name = operation.name
            if "gradients" in operation.name:
                scope_name = name.split("/")[1]
                if scope_name in self.scopes:
                    result[name] = scope_name
                elif scope_name[:-2] in self.scopes:
                    result[name] = scope_name[:-2]
                elif scope_name[:-3] in self.scopes:
                    result[name] = scope_name[:-3]
            else:
                scope_name = name.split("/")[0]
                if scope_name in self.scopes:
                    result[name] = scope_name
                elif scope_name[:-2] in self.scopes:
                    result[name] = scope_name[:-2]
                elif scope_name[:-3] in self.scopes:
                    result[name] = scope_name[:-3]
        for operation in operations:
            if operation.name not in result.keys():
                colocation = operation.colocation_groups()
                lead = colocation[0]
                lead_name = lead.decode().split("@")[1]
                if lead_name not in result.keys():
                    print(operation.name,lead_name)
                    result[operation.name] = self.scopes[0]
                else:
                    result[operation.name] = result[lead_name]

        #check colocation
        '''
        for operation in operations:
            colocation = operation.colocation_groups()
            lead = colocation[0]
            lead_name = lead.decode().split("@")[1]
            if result[lead_name]!=result[operation.name]:
                print("!!!!",operation.name,lead_name,result[operation.name])
            assert(result[lead_name]==result[operation.name])
        '''
        return result
    def build_model(self):
        tf.reset_default_graph()
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
        losses = []
        outputs = []
        tf.get_variable_scope()._reuse =tf.AUTO_REUSE
        for i in range(self.micro_batch_num):
            loss, output, scopes = self.model_fn(None,self.model_name)
            losses.append(loss)
            outputs.append(output[-1])
        self.scopes = scopes
        with tf.variable_scope(self.scopes[-1]):
            new_loss =tf.add_n(losses)
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.2, beta1=0.9, beta2=0.98, epsilon=1e-9).minimize(new_loss)

        init = tf.global_variables_initializer()
        self.graph = tf.get_default_graph()
        self.gdef = tf.get_default_graph().as_graph_def(add_shapes=True)
    def change_model(self,index,config):

        with open( self.model_name+"/"+str(index)+"/init_graph.pbtxt", "w") as f:
            f.write(str(tf.get_default_graph().as_graph_def(add_shapes=True)))

        strategy = {}
        assignment = {}
        for i in range(int(len(config)//2)):
            indexs = config[i*2]
            _strategy = config[i*2+1]
            for j in range(indexs[0],indexs[1]+1,1):
                assignment[self.scopes[j]] = _strategy

        '''
        for i in range(len(self.scopes)):
            if i <8:
                assignment[self.scopes[i]] = [0,0]
            elif i<14:
                assignment[self.scopes[i]] = [1,1]
            else:
                assignment[self.scopes[i]] = [2,3]

        '''
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
        import pickle as pkl
        with open( self.model_name+"/"+str(index)+"/strategy.pkl","wb") as f:
            pkl.dump(strategy,f)
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
        with open( self.model_name+"/"+str(index)+"/modified.pbtxt", "w") as fo:
            fo.write(pbtf.MessageToString(g))

    def activate_unit(self):
        for i in range(1,9,1):
            self.build_model()
            self.change_model(i,four_strategies[i-1])
if __name__ == '__main__':
    config_dict =dict()
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config_dict = json.load(f)
    devices = config_dict.get("devices", [""])

    model_name = config_dict.get("model_name", "bert")



    micro_batch_num =  config_dict.get("micro_batch_num",16)
    batch_size =  config_dict.get("batch_size",6)
    strategy_1 =  config_dict.get("strategy_1",[[0,16],[0,1]])
    strategy_2 =  config_dict.get("strategy_2",[[0,16],[0,1]])
    strategy_3 =  config_dict.get("strategy_3",[[0,16],[0,1]])
    strategy_4 =  config_dict.get("strategy_4",[[0,16],[0,1]])
    strategy_5 =  config_dict.get("strategy_5",[[0,16],[0,1]])
    strategy_6 =  config_dict.get("strategy_6",[[0,16],[0,1]])
    strategy_7 =  config_dict.get("strategy_7",[[0,16],[0,1]])
    strategy_8 =  config_dict.get("strategy_8",[[0,16],[0,1]])

    four_strategies = [strategy_1,strategy_2,strategy_3,strategy_4,strategy_5,strategy_6,strategy_7,strategy_8]

    act = Activater(micro_batch_num = micro_batch_num,batch_size=batch_size,model_name =model_name)
    act.activate_unit()