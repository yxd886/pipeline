import tensorflow as tf
import json
import os

import google.protobuf.text_format as pbtf
import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('../bert/')
sys.path.append('../vgg_19/')
sys.path.append('../resnet152/')
sys.path.append('../inception_v3/')
sys.path.append('../resnet152/')
sys.path.append('../resnet50/')
sys.path.append('../transformer/')
sys.path.append('../xl_net/')



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
            x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
            y = tf.placeholder(tf.float32, shape=(None,1001))
        loss, endpoints,scopes = resnet152_v2.resnet_v2_152(x,y, 1001)
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
        '''
        with tf.variable_scope("input", reuse=tf.AUTO_REUSE):

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

        '''
        tf.get_variable_scope()._reuse =tf.AUTO_REUSE
        for i in range(self.micro_batch_num):
            loss, output, scopes = self.model_fn(None,self.model_name)
            losses.append(loss)
            outputs.append(output[-1])
        self.scopes = scopes
        with tf.variable_scope(self.scopes[-1]):
            new_loss =tf.add_n(losses)/self.micro_batch_num
            new_loss = tf.reduce_mean(new_loss,name="final_loss")
            new_outputs = tf.add_n(outputs)
        #self.train_op = tf.train.AdamOptimizer(learning_rate=0.2, beta1=0.9, beta2=0.98, epsilon=1e-9).minimize(new_loss)
        self.train_op = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(new_loss)

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
            #for i in range(decision[0],decision[1]+1,1):
            for i in decision:
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

        g = (tge.TGE(self.gdef, self.devices)
             .custom(strategy)
             #.replace_placeholder(self.batch_size)
             .use_collective()
             .compile()
             .get_result()
             )
        with open( self.model_name+"/"+str(index)+"/modified.pbtxt", "w") as fo:
            fo.write(pbtf.MessageToString(g))

    def activate_unit(self):
        for i in range(1,len(four_strategies)+1,1):
            self.build_model()
            self.change_model(i,four_strategies[i-1])
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