import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from datasets import dataset_factory
from preprocessing import preprocessing_factory
import tf_slim as slim

dataset = dataset_factory.get_dataset(
    "imagenet", "train", "/data/slim_imagenet")

preprocessing_name = "vgg_19"
image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    preprocessing_name,
    is_training=True)

provider = slim.dataset_data_provider.DatasetDataProvider(
    dataset,
    num_readers=4,
    common_queue_capacity=20 * 32,
    common_queue_min=10 * 32)
[image, label] = provider.get(['image', 'label'])

train_image_size = 224

image = image_preprocessing_fn(image, train_image_size, train_image_size)
print("image shape:", image.shape)
print("label shape:", label.shape)
images, labels = tf.train.batch(
    [image, label],
    batch_size=32,
    num_threads=4,
    capacity=5 * 32)
labels = slim.one_hot_encoding(
    labels, dataset.num_classes)
batch_queue = slim.prefetch_queue.prefetch_queue(
    [images, labels], capacity=2 * 8)
image,label = batch_queue.dequeue()
with open("test_graph.pbtxt", "w") as f:
    f.write(str(tf.get_default_graph().as_graph_def(add_shapes=True)))
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    image,label = sess.run([image,label])
print("aaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(image.shape)
print(label.shape)
