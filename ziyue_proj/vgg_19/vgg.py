# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tf_slim as slim

def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """


def vgg_19(inputs,
           y,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           reuse=None,
           scope='vgg_19',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 19-Layers version E Example.
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)
  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0 or
      None).
    end_points: a dict of tensors with intermediate activations.
  """
  scopes = []
  if True:
    scope_name = tf.get_variable_scope().name
    end_points_collection = tf.get_variable_scope().name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      scopes.append(scope_name+'/conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      scopes.append(scope_name+'/pool1')

      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      scopes.append(scope_name+'/conv2')

      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      scopes.append(scope_name+'/pool2')

      net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
      scopes.append(scope_name+'/conv3')

      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      scopes.append(scope_name+'/pool3')

      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
      scopes.append(scope_name+'/conv4')

      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      scopes.append(scope_name+'/pool4')

      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
      scopes.append(scope_name+'/conv5')

      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      scopes.append(scope_name+'/pool5')


      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      scopes.append(scope_name+'/fc6')

      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      scopes.append(scope_name+'/dropout6')

      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      scopes.append(scope_name+'/fc7')

      # Convert end_points_collection into a end_point dict.
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        scopes.append(scope_name + '/dropout7')

        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        scopes.append(scope_name + '/fc8')
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)


      loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=net)
      loss = tf.reduce_sum(loss)
      return loss, end_points.keys(),end_points.values()
vgg_19.default_image_size = 224