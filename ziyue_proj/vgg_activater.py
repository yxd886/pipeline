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
from datasets import dataset_factory
from preprocessing import preprocessing_factory
import tf_slim as slim

sys.path.append('../')
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



def replace_input(graph,x,name):
    for op in graph.get_operations():
        for i,input in enumerate(op.inputs()):
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
        #server = tf.distribute.Server(cluster, job_name='worker', task_index=0, protocol="grpc+verbs",
         #                                  config=config)
        target = None

        tf.import_graph_def(graph_def)
        print("import success")
        graph = tf.get_default_graph()
        init = graph.get_operation_by_name("import/init/replica_0")
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
            [images, labels], capacity=2 * micro_batch_num)

        sess = tf.Session(target, config=config)  # , config=tf.ConfigProto(allow_soft_placement=False))
        print("222222222222222222222222")

        sess.run(init)
        print("333333333333333333333")


        input_dict = None
        '''
        placeholders = [node.outputs[0] for node in graph.get_operations() if node.node_def.op == 'Placeholder']
        shapes = [(p.shape.as_list()) for p in placeholders ]
        for shape in shapes:
            shape[0]=batch_size
        input_dict = { p: np.random.rand(*shapes[i]) for i,p in enumerate(placeholders) }
        '''
        #prepare input


        x0 = graph.get_tensor_by_name("import/input/Placeholder/replica_0:0")
        x1 = graph.get_tensor_by_name("import/input_1/Placeholder/replica_0:0")
        x2 = graph.get_tensor_by_name("import/input_2/Placeholder/replica_0:0")
        x3 = graph.get_tensor_by_name("import/input_3/Placeholder/replica_0:0")
        x4 = graph.get_tensor_by_name("import/input_4/Placeholder/replica_0:0")
        x5 = graph.get_tensor_by_name("import/input_5/Placeholder/replica_0:0")
        x6 = graph.get_tensor_by_name("import/input_6/Placeholder/replica_0:0")
        x7 = graph.get_tensor_by_name("import/input_7/Placeholder/replica_0:0")
        y0 = graph.get_tensor_by_name("import/input/Placeholder_1/replica_0:0")
        y1 = graph.get_tensor_by_name("import/input_1/Placeholder_1/replica_0:0")
        y2 = graph.get_tensor_by_name("import/input_2/Placeholder_1/replica_0:0")
        y3 = graph.get_tensor_by_name("import/input_3/Placeholder_1/replica_0:0")
        y4 = graph.get_tensor_by_name("import/input_4/Placeholder_1/replica_0:0")
        y5 = graph.get_tensor_by_name("import/input_5/Placeholder_1/replica_0:0")
        y6 = graph.get_tensor_by_name("import/input_6/Placeholder_1/replica_0:0")
        y7 = graph.get_tensor_by_name("import/input_7/Placeholder_1/replica_0:0")

        xs = [x0,x1,x2,x3,x4,x5,x6,x7]
        ys = [y0,y1,y2,y3,y4,y5,y6,y7]

        for i in range(len(xs)):
            x, y = batch_queue.dequeue()
            replace_input(graph,x,xs[i].name)
            replace_input(graph,y,ys[i].name)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        opt = []
        for sink in self.sinks:
            for i in range(10):
                try:
                    op = graph.get_operation_by_name('import/' + sink + "/replica_" + str(i))
                    opt.append(op)
                except:
                    break
        # opt = [graph.get_operation_by_name('import/' + x) for x in self.sinks]
        print("444444444444444444444")

        for j in range(10):  # warm up

            #for i in range(len(xs)):
                #x,y  =batch_queue.dequeue()
                #x, y = sess.run([x,y])
                #input_dict[xs[i]] = x
                #input_dict[ys[i]] = y
                #input_dict[xs[i]] = np.random.rand(32,224,224,3)
                #input_dict[ys[i]] = np.random.rand(32.1001)

            sess.run(opt, feed_dict=input_dict)


        times= []
        for j in range(10):
            tmp = time.time()

            #for i in range(len(xs)):
                #x,y  =batch_queue.dequeue()
                #x, y = sess.run([x,y])
                #input_dict[xs[i]] = x
                #input_dict[ys[i]] = y
                #input_dict[xs[i]] = np.random.rand(32,224,224,3)
                #input_dict[ys[i]] = np.random.rand(32.1001)

            sess.run(opt, feed_dict=input_dict)
            times.append(time.time()-tmp)
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