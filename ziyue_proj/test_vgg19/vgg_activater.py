import numpy as np
import tensorflow as tf
import json
import os
import time
import sys

sys.path.append('../')
sys.path.append('../../')
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import step_stats_pb2
import google.protobuf.text_format as pbtf
import pickle as pkl
from tensorflow.python.client import timeline
from tensorflow.distribute.cluster_resolver import TFConfigClusterResolver
from datasets import dataset_factory
from preprocessing import preprocessing_factory
import tf_slim as slim


import multiprocessing as mp
arg_prefix=sys.argv[1]

config_dict =dict()
if os.path.exists("vgg_config.json"):
    with open("vgg_config.json", "r") as f:
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
model_name = config_dict.get("model_name", "bert")
activate_graph=config_dict.get("activate_graph", "1")
activate_graphs = [model_name+"/"+activate_graph+"/modified.pbtxt"]
sinks = config_dict.get("activate_sink", ["GradientDescent"])
micro_batch_num = config_dict.get("micro_batch_num", 8)
batch_size = config_dict.get("batch_size", 32)
model_name = config_dict.get("model_name", "bert")
global_batch_size = batch_size*micro_batch_num


def get_tensors(graph,name):
    ret = []
    for op in graph.get_operations():
        for tensor in op.outputs:
            if name in tensor.name and "gradient" not in tensor.name:
                ret.append(tensor)
    return ret


def get_one_tensor(graph,name):
    for op in graph.get_operations():
        for tensor in op.outputs:
            if name in tensor.name and "gradient" not in tensor.name:
                return tensor

def replace_input(graph,x,name):
    for op in graph.get_operations():
        for i,input in enumerate(op.inputs):
            if input.name==name:
                op._update_input(i,x)


class Activater():
    def __init__(self, activate_path, sinks=["Adam"]):
        self.graph_defs = []
        self.path = []
        for path in  activate_path:
            if os.path.exists(path):
                self.path.append(path)
        for path in self.path:
            gdef = graph_pb2.GraphDef()
            with open(path,"r")as f:
                txt = f.read()
            pbtf.Parse(txt,gdef)
            self.graph_defs.append(gdef)

        self.sinks = sinks
        self.server=None

    def activate_unit(self,path,graph_def):
        #setup_workers(workers, "grpc+verbs")
        tf.reset_default_graph()

        #server = tf.distribute.Server(cluster, job_name='worker', task_index=0, protocol="grpc+verbs",
         #                                  config=config)
        target = None

        tf.import_graph_def(graph_def)
        print("import success")
        graph = tf.get_default_graph()
        init0 = graph.get_operation_by_name("import/init/replica_0")
        print("11111111111111111111111")

        dataset = dataset_factory.get_dataset(
            "imagenet", "train", "/data/slim_imagenet")

        preprocessing_name = "vgg_19"
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True)

        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=4,
            common_queue_capacity=20 * batch_size*micro_batch_num,
            common_queue_min=10 * batch_size*micro_batch_num,)
        [image, label] = provider.get(['image', 'label'])

        train_image_size = 224


        image = image_preprocessing_fn(image, train_image_size, train_image_size)
        print("image shape:", image.shape)
        print("label shape:", label.shape)
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size*micro_batch_num,
            num_threads=4,
            capacity=5 * batch_size*micro_batch_num)
        labels = slim.one_hot_encoding(
            labels, dataset.num_classes)
        batch_queue = slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=2 * micro_batch_num)

        input_dict = None
        '''
        placeholders = [node.outputs[0] for node in graph.get_operations() if node.node_def.op == 'Placeholder']
        shapes = [(p.shape.as_list()) for p in placeholders ]
        for shape in shapes:
            shape[0]=batch_size
        input_dict = { p: np.random.rand(*shapes[i]) for i,p in enumerate(placeholders) }
        '''
        #prepare input

        xs = ["import/input/Placeholder/replica_0:0"]
        ys = ["import/input/Placeholder_1/replica_0:0"]
        for i in range(1,micro_batch_num):
            xs.append("import/input_{}/Placeholder/replica_0:0".format(i))
            ys.append("import/input_{}/Placeholder_1/replica_0:0".format(i))
        x, y = batch_queue.dequeue()
        for i in range(len(xs)):
            replace_input(graph,x[i*batch_size:(i+1)*batch_size],xs[i])
            replace_input(graph,y[i*batch_size:(i+1)*batch_size],ys[i])
        losses = get_tensors(graph, "final_loss")
        losses = tf.reduce_mean(tf.add_n(losses)/len(losses))
        accurate_num = get_tensors(graph,"top_accuracy")
        print("accurate_num:",accurate_num)
        total_batch_size = batch_size*micro_batch_num
        size_for_each = total_batch_size/len(accurate_num)
        num_to_calculate = int(64/size_for_each)
        accurate_num = tf.reduce_sum(tf.add_n(accurate_num[:num_to_calculate]))

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        sess = tf.Session(target, config=config)  # , config=tf.ConfigProto(allow_soft_placement=False))
        print("222222222222222222222222")
        print("333333333333333333333")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        opt = []
        for sink in self.sinks:
            op = graph.get_operation_by_name('import/' + sink + "/replica_0")
            opt.append(op)
        # opt = [graph.get_operation_by_name('import/' + x) for x in self.sinks]
        print("444444444444444444444")
        recorded_accuracy5 = []
        global_start_time = time.time()
        with open("time_record.txt", "w") as f:
            f.write("global start time: {}\n".format(global_start_time))
        times= []

        sess.run(init0)
        #sess.run(init1)

        start_time = time.time()
        for j in range(100000000000000):
            ret = sess.run(opt + [losses,accurate_num], feed_dict=input_dict)
            loss = ret[-2]
            top5accuracy_num = ret[-1]
            top5accuracy = top5accuracy_num/64
            if j % 10 == 0:
                end_time = time.time()
                print("Step:{},Loss:{},top5 accuracy:{},per_step_time:{}".format(j,loss,top5accuracy,(end_time-start_time)/10))
                start_time = time.time()
            gap = top5accuracy*100 // 5 * 5
            if gap not in recorded_accuracy5:
                global_end_time = time.time()
                recorded_accuracy5.append(gap)
                print("achieveing {}% at the first time, concreate top5 accuracy: {}%. time slot: {}, duration: {}s\n".format(gap,top5accuracy*100,global_end_time,global_end_time-global_start_time),flush=True)
                with open("time_record.txt","a+") as f:
                    f.write("achieveing {}% at the first time, concreate top5 accuracy: {}%. time slot: {}, duration: {}s\n".format(gap,top5accuracy*100,global_end_time,global_end_time-global_start_time))




        avg_time = sum(times)/len(times)
        print(path,times,"average time:", avg_time)
        print(" ")
        '''
        if arg_prefix=="profile":
            for i in range(10):
                run_meta = tf.RunMetadata()
                run_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
                sess.run(opt, feed_dict=input_dict,
                         options=run_opt,
                         run_metadata=run_meta
                         )
                tf.contrib.tfprof.model_analyzer.print_model_analysis(tf.get_default_graph(),run_meta=run_meta,tfprof_options =tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)
            tl = timeline.Timeline(run_meta.step_stats)
        else:
            for i in range(1):
                run_meta = tf.RunMetadata()
                run_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
                sess.run(opt, feed_dict=input_dict,
                         options=run_opt,
                         run_metadata=run_meta
                         )
                #tf.contrib.tfprof.model_analyzer.print_model_analysis(tf.get_default_graph(),run_meta=run_meta,tfprof_options =tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)
            tl = timeline.Timeline(run_meta.step_stats)

        with open(path.split(".")[0] + "_timeline.json", "w") as fo:
            fo.write(tl.generate_chrome_trace_format())
        with open(path.split(".")[0] + "_runmeta_.pbtxt", "w") as fo:
            fo.write(pbtf.MessageToString(run_meta))
        '''
    def activate(self):
        for k,graph_def in enumerate(self.graph_defs):
            self.activate_unit(self.path[k],graph_def)

workers = config_dict.get("workers", ["10.28.1.26:3901","10.28.1.17:3901","10.28.1.16:3901"])

clus = dict()
clus["cluster"] = {"worker": workers}
clus["task"] = {"type": "worker", "index": 0}
os.environ["TF_CONFIG"] = json.dumps(clus)


act = Activater(activate_graphs,sinks=sinks)
act.activate()
a = input("enter to exit")