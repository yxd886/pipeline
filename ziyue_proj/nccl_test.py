import sys
import tensorflow as tf
from tensorflow.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.ops import collective_ops
tf.logging.set_verbosity('DEBUG')

def test_dist():
    ts = []
    for task_id in (0, 1, 2):
        with tf.device('/job:worker/task:{0}/device:GPU:0'.format(task_id)):
            t = tf.Variable([1.0,3.0*task_id], dtype=tf.float32, name='myvar')
            ts.append(t)

    with tf.device('/job:worker/task:0/device:GPU:0'):
        sum0 = collective_ops.all_reduce(ts[0], 2, 0, 1, 'Add', 'Id')
    with tf.device('/job:worker/task:1/device:GPU:0'):
        sum1 = collective_ops.all_reduce(ts[1], 2, 0, 1, 'Add', 'Id')

    # with tf.control_dependencies([sum0, sum1]):
    with tf.device('/job:worker/task:0/device:GPU:0'):
        sumb0 = collective_ops.all_reduce(tf.identity(ts[0]), 3, 1, 2, 'Add', 'Id')
    with tf.device('/job:worker/task:1/device:GPU:0'):
        sumb1 = collective_ops.all_reduce(tf.identity(ts[1]), 3, 1, 2, 'Add', 'Id')
    with tf.device('/job:worker/task:2/device:GPU:0'):
        sumb2 = collective_ops.all_reduce(ts[0], 3, 1, 2, 'Add', 'Id')

    resolver = TFConfigClusterResolver()
    cluster = resolver.cluster_spec()

    dist = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.NCCL)

    sess_config = dist.update_config_proto(tf.ConfigProto())
    sess_config.ClearField("device_filters")

    server = tf.distribute.Server(
        cluster, job_name="worker", task_index=0, config=sess_config)

    sess = tf.compat.v1.Session(server.target, config=sess_config)
    sess.run(tf.compat.v1.global_variables_initializer())

    print('tensor value', sess.run([sum0, sum1, sumb0, sumb1, sumb2]))

    with open("graph_def", "w") as f:
        f.write(str(tf.get_default_graph().as_graph_def()))

import os
import json
config_dict =dict()
if os.path.exists("config.json"):
    with open("config.json", "r") as f:
        config_dict = json.load(f)

workers = config_dict.get("workers", ["10.28.1.26:3901","10.28.1.17:3901","10.28.1.16:3901"])

clus = dict()
clus["cluster"] = {"worker": workers}
clus["task"] = {"type": "worker", "index": 0}
os.environ["TF_CONFIG"] = json.dumps(clus)


id = int(sys.argv[1])
if id == 0:
    test_dist()
else:
    resolver = TFConfigClusterResolver()
    cluster = resolver.cluster_spec()
    dist = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.NCCL)
    sess_config = dist.update_config_proto(tf.ConfigProto())
    sess_config.ClearField("device_filters")
    server = tf.distribute.Server(
        cluster, job_name="worker", task_index=id, config=sess_config)
    server.join()