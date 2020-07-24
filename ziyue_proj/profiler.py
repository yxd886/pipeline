import numpy as np
import tensorflow as tf
import re
import itertools
import os
import json
from pipeline import model_fn


config_dict =dict()
if os.path.exists("config.json"):
    with open("config.json", "r") as f:
        config_dict = json.load(f)
batch_size = config_dict.get("batch_size", 6)
model_name = config_dict.get("model_name", "bert")

with tf.device("/device:GPU:0"):
    loss,output,scopes = model_fn(batch_size,model_name)
    #vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #gradients = tf.compat.v1.gradients(loss, vars, colocate_gradients_with_ops=True)
    train_op =tf.train.GradientDescentOptimizer(0.2).minimize(loss)
    init_op = tf.global_variables_initializer()
    graph =tf.get_default_graph()
    placeholders = [node.outputs[0] for node in graph.get_operations() if node.node_def.op == 'Placeholder']
    shapes = [(p.shape.as_list()) for p in placeholders]
    for shape in shapes:
        shape[0] = batch_size
    input_dict = {p: np.random.rand(*shapes[i]) for i, p in enumerate(placeholders)}
config = tf.ConfigProto()
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(init_op)
import time
run_meta = tf.compat.v1.RunMetadata()
run_opt = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)  # , output_partition_graphs=True)
for i in range(10):#warm up
    sta = time.time()
    sess.run([train_op,loss], feed_dict=input_dict)
    print("time:",time.time()-sta)
sess.run([train_op,loss], options=run_opt, run_metadata=run_meta, feed_dict=input_dict)

result = {}
layer_result = {item:[0,0,0,0] for item in scopes}
times = {}
names = []
scope_name_dict = {item:[] for item in scopes}
with open("graph.pbtxt","w") as f:
    f.write(str(tf.get_default_graph().as_graph_def(add_shapes=True)))
for dev in run_meta.step_stats.dev_stats:
    if 'Kernel' not in dev.device and 'stream' not in dev.device:  # TODO: if no GPU data for this op, use the CPU data
        continue
    for node in dev.node_stats:
        name = node.node_name.split(':')[0]
        if name not in result:
            result[name] = [float('inf'), 0]
        result[name][0] = min(result[name][0], node.all_start_micros)
        result[name][1] = max(result[name][1], node.all_start_micros + node.all_end_rel_micros)
        if name not in names:
            names.append(name)


# layer time
print(scopes)
for name in names:
    if "gradients" in name:
        scope = name.split("/")[1]
        if scope not in scopes:
            print(scope)
            continue
        scope_name_dict[scope].append(name)
        if scope not in layer_result:
            layer_result[scope] = [0, 0, 0, 0]
        layer_result[scope][1] += (result[name][1] - result[name][0])
    else:
        scope = name.split("/")[0]
        if scope not in scopes:
            print(scope)
            continue
        scope_name_dict[scope].append(name)
        if scope not in layer_result:
            layer_result[scope] = [0, 0, 0, 0]
        layer_result[scope][0] += (result[name][1] - result[name][0])


#parameter size
for scope in scopes:
    total_parameters = 0
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)
    for variable in vars:
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    layer_result[scope][3] = total_parameters*4


#activation size
assert(len(output)==len(scopes))
for i,tensor in enumerate(output):
    shape = tensor.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    scope_name = scopes[i]
    if i==0:
        layer_result[scope_name][2] = variable_parameters * 4*3
    else:
        layer_result[scope_name][2] = variable_parameters*4

#for i,layer_name in enumerate(layer_result):
#    index = order_scopes.index(layer_name)
#    if index==0:
#        continue
#    layer_result[layer_name][0] = max(layer_result[layer_name][0],layer_result[order_scopes[index-1]][1])


import json
with open("profile.json","w") as f:
    json.dump(layer_result,f,indent=2,sort_keys=True)

with open("scope_name_dict.json","w") as f:
    json.dump(scope_name_dict,f,indent=2,sort_keys=True)


graph = tf.get_default_graph()
operation_names = [item.name for item in graph.get_operations()]


contents = []
for i,scope in enumerate(scopes):
    contents.append("node{} -- {} -- forward_compute_time={}, backward_compute_time={}, activation_size={}, parameter_size={}\n".format(i,scope,layer_result[scope][0],layer_result[scope][1],layer_result[scope][2],layer_result[scope][3]))
for i in range(len(scopes)-1):
    contents.append("\tnode{} -- node{}\n".format(i,i+1))
with open("graph.txt","w") as f:
    f.writelines(contents)


import json
with open("run_meta.pbtxt","w") as f:
    f.write(str(run_meta))

'''
result = {}
names = [item.op.name for item in output]
reverse_names = find_bp_point(graph,names,operation_names)
for dev in run_meta.step_stats.dev_stats:
    for node in dev.node_stats:
        name = node.node_name
        if ":" in name:
            name = name.split(':')[0]
        if name in names:
            result[name] = node.all_start_micros + node.all_end_rel_micros

print(len(result),len(names))
assert(len(result)==len(names))
for dev in run_meta.step_stats.dev_stats:
    for node in dev.node_stats:
        name = node.node_name
        if ":" in name:
            name = name.split(':')[0]
        if name in reverse_names:
            result[name] = node.all_start_micros + node.all_end_rel_micros

print(len(result),len(reverse_names))
assert(len(result)==len(names)+len(reverse_names))
with open("profile1.json","w") as f:
    json.dump(result,f,indent=2,sort_keys=True)

'''