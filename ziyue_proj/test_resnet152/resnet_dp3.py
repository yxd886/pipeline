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
sys.path.append('../../')
from datasets import dataset_factory
from preprocessing import preprocessing_factory
import tf_slim as slim

sys.path.append('../bert/')
sys.path.append('../vgg_19/')
sys.path.append('../resnet152/')
sys.path.append('../inception_v3/')
sys.path.append('../resnet152/')
sys.path.append('../resnet50/')
sys.path.append('../transformer/')
sys.path.append('../xl_net/')
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


def model_fn(batch_queue,model_name):

    if model_name=="resnet152":
        import resnet152_v2
        with tf.variable_scope("input", reuse=tf.AUTO_REUSE):
            #x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
            #y = tf.placeholder(tf.float32, shape=(None,1,1,1001))
            x,y = batch_queue.dequeue()
        loss, endpoints,scopes = resnet152_v2.resnet_v2_152(x,y, 1001)
        return loss, [x] + endpoints, ["input"] + scopes


def get_tensors(graph,name):
    ret = []
    for op in graph.get_operations():
        for tensor in op.outputs:
            if name in tensor.name and "gradient" not in tensor.name:
                ret.append(tensor)
    return ret

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

        gpu_num = 4

        recorded_accuracy5 = []
        global_start_time = time.time()
        with open("resnet_dp3_time_record.txt", "w") as f:
            f.write("global start time: {}\n".format(global_start_time))
        times= []

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

        with tf.variable_scope("input", reuse=tf.AUTO_REUSE):

            dataset = dataset_factory.get_dataset(
                "imagenet", "train", "/data/slim_imagenet")

            preprocessing_name = "resnet_v2_152"
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                preprocessing_name,
                is_training=True)

            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=4,
                common_queue_capacity=20 * batch_size,
                common_queue_min=10 * batch_size)
            [image, label] = provider.get(['image', 'label'])

            train_image_size = 224

            image = image_preprocessing_fn(image, train_image_size, train_image_size)
            print("image shape:", image.shape)
            print("label shape:", label.shape)
            images, labels = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=4,
                capacity=5 * batch_size)
            labels = slim.one_hot_encoding(
                labels, dataset.num_classes)
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=2 * gpu_num)


        tf.get_variable_scope()._reuse =tf.AUTO_REUSE
        for i in range(gpu_num):
            with tf.device("gpu:{}".format(i)):
                loss, output, scopes = self.model_fn(batch_queue,self.model_name)
                losses.append(loss)
                outputs.append(output[-1])
        self.scopes = scopes
        with tf.device("gpu:2"):
            new_loss =tf.add_n(losses,name="final_loss")/gpu_num
            new_loss = tf.reduce_mean(new_loss)
            new_outputs = tf.add_n(outputs)
        #self.train_op = tf.train.AdamOptimizer(learning_rate=0.2, beta1=0.9, beta2=0.98, epsilon=1e-9).minimize(new_loss)
            #self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(new_loss,colocate_gradients_with_ops=True)
        self.train_op =tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(new_loss,colocate_gradients_with_ops=True)

        graph = tf.get_default_graph()
        accurate_num = get_tensors(graph,"top_accuracy")
        print("accurate_num:",accurate_num)
        #accurate_num = tf.reduce_sum(tf.add_n(accurate_num))
        accurate_num = tf.reduce_sum(accurate_num[0])



        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        start_time = time.time()

        accurate_times = []
        total_times = []
        for i in range(10000000):
            _,loss,accuracy_num = sess.run([self.train_op,new_loss,accurate_num])
            #top5accuracy = accuracy_num / (gpu_num * batch_size)
            accurate_times.append(accuracy_num)
            total_times.append(batch_size)


            if i%10==0:
                end_time = time.time()
                print("Step:{},Loss:{},per_step_time:{}".format(i,loss,(end_time-start_time)/10))
                start_time = time.time()

            if sum(total_times)==64:
                top5accuracy = sum(accurate_times) / 64
                print("Step:{},Loss:{},accuracy:{}".format(i,loss,top5accuracy))
                accurate_times = []
                total_times = []


                gap = top5accuracy*100 // 5 * 5
                if gap not in recorded_accuracy5:
                    global_end_time = time.time()
                    recorded_accuracy5.append(gap)
                    print("achieveing {}% at the first time, concreate top5 accuracy: {}%. time slot: {}, duration: {}s\n".format(gap,top5accuracy*100,global_end_time,global_end_time-global_start_time),flush=True)
                    with open("resnet_dp3_time_record.txt","a+") as f:
                        f.write("achieveing {}% at the first time, concreate top5 accuracy: {}%. time slot: {}, duration: {}s\n".format(gap,top5accuracy*100,global_end_time,global_end_time-global_start_time))




    def activate_unit(self):
        self.build_model()
if __name__ == '__main__':
    config_dict =dict()
    if os.path.exists("resnet_config.json"):
        with open("resnet_config.json", "r") as f:
            config_dict = json.load(f)
    devices = config_dict.get("devices", [""])

    model_name = config_dict.get("model_name", "resnet152")



    micro_batch_num =  config_dict.get("micro_batch_num",8)
    batch_size =  config_dict.get("batch_size",32)
    strategy_1 =  config_dict.get("strategy_1",[[0,16],[0,1]])
    strategy_2 =  config_dict.get("strategy_2",[[0,16],[0,1]])
    strategy_3 =  config_dict.get("strategy_3",[[0,16],[0,1]])
    strategy_4 =  config_dict.get("strategy_4",[[0,16],[0,1]])
    pipedream = config_dict.get("pipedream", strategy_1)
    hetpipe =config_dict.get("hetpipe",strategy_1)
    gpipe =config_dict.get("gpipe",strategy_1)

    four_strategies = [strategy_1,strategy_2,strategy_3,strategy_4,pipedream,hetpipe,gpipe]

    act = Activater(micro_batch_num = micro_batch_num,batch_size=batch_size,model_name =model_name)
    act.activate_unit()