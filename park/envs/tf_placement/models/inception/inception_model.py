# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inception-v3 expressed in TensorFlow-Slim.

  Usage:

  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the batch_norm moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      # Force all Variables to reside on the CPU.
      with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
        logits, endpoints = slim.inception.inception_v3(
            images,
            dropout_keep_prob=0.8,
            num_classes=num_classes,
            is_training=for_training,
            restore_logits=restore_logits,
            scope=scope)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops

sys.path.append('park/envs/tf_placement/models/inception')
import scopes
from park.envs.tf_placement.models.inception import ops
import os
import random
sys.path.append('park/envs/tf_placement/models/')
from custom_tf_scope import CustomTFScope
from nn_model import *

class InceptionV3(NNModel):

    def build(self, inputs=None,
                    dropout_keep_prob=0.8,
                    num_classes=1000,
                    trainable=True,
                    restore_logits=True,
                    bs=128,
                    scope='',
                    devices=None,
                    device_placement=None):
        """Latest Inception from http://arxiv.org/abs/1512.00567.

            "Rethinking the Inception Architecture for Computer Vision"

            Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
            Zbigniew Wojna

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            dropout_keep_prob: dropout keep_prob.
            num_classes: number of predicted classes.
            is_training: whether is training or not.
            restore_logits: whether or not the logits layers should be restored.
            Useful for fine-tuning a model with different num_classes.
            scope: Optional scope for name_scope.

        Returns:
            a list containing 'logits', 'aux_logits' Tensors.
        """
        # end_points will collect relevant activations for external use, for example
        # summaries or losses.


        def get_dev_id(i, j=None):
            N = len(devices)
            if device_placement == 'random':
                dev_id = random.randint(0, N-1)
            elif device_placement == 'alternate':
                dev_id = i% N
            elif device_placement == 'mem_efficient':
                if j is None:
                    dev_id = i% N
                else:
                    dev_id = j% N

                print(dev_id)
            else:
                dev_id = 0
            return dev_id

        g = tf.Graph()
        with g.as_default():
            with CustomTFScope('dummy_input'):
                if inputs is None:
                    inputs = tf.ones((bs, 299, 299, 3))

            end_points = {}
            with tf.name_scope(scope, 'inception_v3', [inputs]):
                with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                                    is_training=trainable):
                    with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                        with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                                                stride=1, padding='VALID'):

                            with tf.device(devices[get_dev_id(0, 0)]) if devices else ExitStack() as gs:
                                with CustomTFScope('conv0'):
                                    # 299 x 299 x 3
                                    end_points['conv0'] = ops.conv2d(inputs, 32, [3, 3], stride=2)

                            with tf.device(devices[get_dev_id(0, 1)]) if devices else ExitStack() as gs:
                                with CustomTFScope('conv1'):
                                    # 149 x 149 x 32
                                    end_points['conv1'] = ops.conv2d(end_points['conv0'], 32, [3, 3])

                            with tf.device(devices[get_dev_id(0, 2)]) if devices else ExitStack() as gs:
                                with CustomTFScope('conv2'):
                                    # 147 x 147 x 32
                                    end_points['conv2'] = ops.conv2d(end_points['conv1'], 64, [3, 3])

                            with tf.device(devices[get_dev_id(0, 3)]) if devices else ExitStack() as gs:
                                with CustomTFScope('pool1'):
                                    # 147 x 147 x 64
                                    end_points['pool1'] = ops.max_pool(end_points['conv2'], [3, 3],
                                                            stride=2)

                            with tf.device(devices[get_dev_id(0, 3)]) if devices else ExitStack() as gs:
                                with CustomTFScope('conv3'):
                                    # 73 x 73 x 64
                                    end_points['conv3'] = ops.conv2d(end_points['pool1'], 80, [1, 1])

                            with tf.device(devices[get_dev_id(0, 4)]) if devices else ExitStack() as gs:
                                with CustomTFScope('conv4'):
                                    # 73 x 73 x 80.
                                    end_points['conv4'] = ops.conv2d(end_points['conv3'], 192, [3, 3])
                                    
                            with tf.device(devices[get_dev_id(0, 5)]) if devices else ExitStack() as gs:
                                with CustomTFScope('pool2'):
                                    # 71 x 71 x 192.
                                    end_points['pool2'] = ops.max_pool(end_points['conv4'], [3, 3],
                                                            stride=2)
                            # 35 x 35 x 192.
                            net = end_points['pool2']
                    # Inception blocks
                    with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                                        stride=1, padding='SAME'):
                        # mixed: 35 x 35 x 256.
                        with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                            with CustomTFScope('mixed_35x35x256a'):
                                with tf.device(devices[get_dev_id(1)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch1x1'):
                                        branch1x1 = ops.conv2d(net, 64, [1, 1])
                                with tf.device(devices[get_dev_id(2)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch5x5'):
                                        with CustomTFScope('conv_0'):
                                            branch5x5 = ops.conv2d(net, 48, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
                                with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch3x3dbl'):
                                        with CustomTFScope('conv_0'):
                                            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                                        with CustomTFScope('conv_2'):
                                            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                                with tf.device(devices[get_dev_id(3)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch_pool'):
                                        with CustomTFScope('avg_pool'):
                                            branch_pool = ops.avg_pool(net, [3, 3])
                                        with CustomTFScope('conv_0'):
                                            branch_pool = ops.conv2d(branch_pool, 32, [1, 1])
                                net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
                                end_points['mixed_35x35x256a'] = net
                        # mixed_1: 35 x 35 x 288.
                        with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                            with CustomTFScope('mixed_35x35x288a'):
                                with tf.device(devices[get_dev_id(1)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch1x1'):
                                        branch1x1 = ops.conv2d(net, 64, [1, 1])
                                with tf.device(devices[get_dev_id(2)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch5x5'):
                                        with CustomTFScope('conv_0'):
                                            branch5x5 = ops.conv2d(net, 48, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
                                with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch3x3dbl'):
                                        with CustomTFScope('conv_0'):
                                            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                                        with CustomTFScope('conv_2'):
                                            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                                with tf.device(devices[get_dev_id(3)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch_pool'):
                                        with CustomTFScope('avg_pool'):
                                            branch_pool = ops.avg_pool(net, [3, 3])
                                        with CustomTFScope('conv_0'):
                                            branch_pool = ops.conv2d(branch_pool, 64, [1, 1])
                                net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
                                end_points['mixed_35x35x288a'] = net
                        # mixed_2: 35 x 35 x 288.
                        with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                            with CustomTFScope('mixed_35x35x288b'):
                                with CustomTFScope('branch1x1'):
                                    with tf.device(devices[get_dev_id(1)]) if devices else ExitStack() as gs:
                                        branch1x1 = ops.conv2d(net, 64, [1, 1])
                                with tf.device(devices[get_dev_id(2)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch5x5'):
                                        with CustomTFScope('conv_0'):
                                            branch5x5 = ops.conv2d(net, 48, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
                                with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch3x3dbl'):
                                        with CustomTFScope('conv_0'):
                                            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                                        with CustomTFScope('conv_2'):
                                            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                                with tf.device(devices[get_dev_id(3)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch_pool'):
                                        with CustomTFScope('avg_pool'):
                                            branch_pool = ops.avg_pool(net, [3, 3])
                                        with CustomTFScope('conv_0'):
                                            branch_pool = ops.conv2d(branch_pool, 64, [1, 1])
                                net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
                                end_points['mixed_35x35x288b'] = net
                        # mixed_3: 17 x 17 x 768.
                        with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                            with CustomTFScope('mixed_17x17x768a'):
                                with tf.device(devices[get_dev_id(1)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch3x3'):
                                        branch3x3 = ops.conv2d(net, 384, [3, 3], stride=2, padding='VALID')
                                with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch3x3dbl'):
                                        with CustomTFScope('conv_0'):
                                            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                                        with CustomTFScope('conv_2'):
                                            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],
                                                            stride=2, padding='VALID')
                                with tf.device(devices[get_dev_id(2)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch_pool'):
                                        branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
                                net = tf.concat(axis=3, values=[branch3x3, branch3x3dbl, branch_pool])
                                end_points['mixed_17x17x768a'] = net
                        # mixed4: 17 x 17 x 768.
                        with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                            with CustomTFScope('mixed_17x17x768b'):
                                with tf.device(devices[get_dev_id(1)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch1x1'):
                                        branch1x1 = ops.conv2d(net, 192, [1, 1])
                                with tf.device(devices[get_dev_id(2)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch7x7'):
                                        with CustomTFScope('conv_0'):
                                            branch7x7 = ops.conv2d(net, 128, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch7x7 = ops.conv2d(branch7x7, 128, [1, 7])
                                        with CustomTFScope('conv_2'):
                                            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
                                with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch7x7dbl'):
                                        with CustomTFScope('conv_0'):
                                            branch7x7dbl = ops.conv2d(net, 128, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
                                        with CustomTFScope('conv_2'):
                                            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [1, 7])
                                        with CustomTFScope('conv_3'):
                                            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
                                        with CustomTFScope('conv_4'):
                                            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
                                with tf.device(devices[get_dev_id(3)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch_pool'):
                                        with CustomTFScope('avg_pool'):
                                            branch_pool = ops.avg_pool(net, [3, 3])
                                        with CustomTFScope('conv_0'):
                                            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                                net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
                                end_points['mixed_17x17x768b'] = net
                        # mixed_5: 17 x 17 x 768.
                        with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                            with CustomTFScope('mixed_17x17x768c'):
                                with tf.device(devices[get_dev_id(1)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch1x1'):
                                        branch1x1 = ops.conv2d(net, 192, [1, 1])
                                with tf.device(devices[get_dev_id(2)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch7x7'):
                                        with CustomTFScope('conv_0'):
                                            branch7x7 = ops.conv2d(net, 160, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch7x7 = ops.conv2d(branch7x7, 160, [1, 7])
                                        with CustomTFScope('conv_2'):
                                            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
                                with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch7x7dbl'):
                                        with CustomTFScope('conv_0'):
                                            branch7x7dbl = ops.conv2d(net, 160, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
                                        with CustomTFScope('conv_2'):
                                            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [1, 7])
                                        with CustomTFScope('conv_3'):
                                            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
                                        with CustomTFScope('conv_4'):
                                            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
                                with tf.device(devices[get_dev_id(3)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch_pool'):
                                        with CustomTFScope('avg_pool'):
                                            branch_pool = ops.avg_pool(net, [3, 3])
                                        with CustomTFScope('conv_0'):
                                            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                                net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
                                end_points['mixed_17x17x768c'] = net
                        # mixed_6: 17 x 17 x 768.
                        with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                            with CustomTFScope('mixed_17x17x768d'):
                                with tf.device(devices[get_dev_id(1)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch1x1'):
                                        branch1x1 = ops.conv2d(net, 192, [1, 1])
                                with tf.device(devices[get_dev_id(2)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch7x7'):
                                        with CustomTFScope('conv_0'):
                                            branch7x7 = ops.conv2d(net, 160, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch7x7 = ops.conv2d(branch7x7, 160, [1, 7])
                                        with CustomTFScope('conv_2'):
                                            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
                                with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch7x7dbl'):
                                        with CustomTFScope('conv_0'):
                                            branch7x7dbl = ops.conv2d(net, 160, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
                                        with CustomTFScope('conv_2'):
                                            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [1, 7])
                                        with CustomTFScope('conv_3'):
                                            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
                                        with CustomTFScope('conv_4'):
                                            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
                                with tf.device(devices[get_dev_id(3)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch_pool'):
                                        with CustomTFScope('avg_pool'):
                                            branch_pool = ops.avg_pool(net, [3, 3])
                                        with CustomTFScope('conv_0'):
                                            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                                net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
                                end_points['mixed_17x17x768d'] = net
                        # mixed_7: 17 x 17 x 768.
                        with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                            with CustomTFScope('mixed_17x17x768e'):
                                with tf.device(devices[get_dev_id(1)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch1x1'):
                                        branch1x1 = ops.conv2d(net, 192, [1, 1])
                                with tf.device(devices[get_dev_id(2)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch7x7'):
                                        with CustomTFScope('conv_0'):
                                            branch7x7 = ops.conv2d(net, 192, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch7x7 = ops.conv2d(branch7x7, 192, [1, 7])
                                        with CustomTFScope('conv_2'):
                                            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
                                with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch7x7dbl'):
                                        with CustomTFScope('conv_0'):
                                            branch7x7dbl = ops.conv2d(net, 192, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [7, 1])
                                        with CustomTFScope('conv_2'):
                                            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
                                        with CustomTFScope('conv_3'):
                                            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [7, 1])
                                        with CustomTFScope('conv_4'):
                                            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
                                with tf.device(devices[get_dev_id(3)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch_pool'):
                                        with CustomTFScope('avg_pool'):
                                            branch_pool = ops.avg_pool(net, [3, 3])
                                        with CustomTFScope('conv_0'):
                                            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                                net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
                                end_points['mixed_17x17x768e'] = net
                        # Auxiliary Head logits
                        with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                            aux_logits = tf.identity(end_points['mixed_17x17x768e'])
                            with CustomTFScope('aux_logits'):
                                with CustomTFScope('avg_pool'):
                                    aux_logits = ops.avg_pool(aux_logits, [5, 5], stride=3,
                                                            padding='VALID')
                                with CustomTFScope('conv_0'):
                                    aux_logits = ops.conv2d(aux_logits, 128, [1, 1], scope='proj')
                                # Shape of feature map before the final layer.
                                shape = aux_logits.get_shape()
                                with CustomTFScope('conv_1'):
                                    aux_logits = ops.conv2d(aux_logits, 768, shape[1:3], stddev=0.01,
                                                        padding='VALID')
                                    aux_logits = ops.flatten(aux_logits)
                                with CustomTFScope('fc'):
                                    aux_logits = ops.fc(aux_logits, num_classes, activation=None,
                                                    stddev=0.001, restore=restore_logits)
                                end_points['aux_logits'] = aux_logits
                        # mixed_8: 8 x 8 x 1280.
                        # Note that the scope below is not changed to not void previous
                        # checkpoints.
                        # (TODO) Fix the scope when appropriate.
                        with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                            with CustomTFScope('mixed_17x17x1280a'):
                                with tf.device(devices[get_dev_id(1)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch3x3'):
                                        with CustomTFScope('conv_0'):
                                            branch3x3 = ops.conv2d(net, 192, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch3x3 = ops.conv2d(branch3x3, 320, [3, 3], stride=2,
                                                            padding='VALID')
                                with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch7x7x3'):
                                        with CustomTFScope('conv_0'):
                                            branch7x7x3 = ops.conv2d(net, 192, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [1, 7])
                                        with CustomTFScope('conv_2'):
                                            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [7, 1])
                                        with CustomTFScope('conv_3'):
                                            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [3, 3],
                                                                stride=2, padding='VALID')
                                with tf.device(devices[get_dev_id(2)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch_pool'):
                                        branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
                                net = tf.concat(axis=3, values=[branch3x3, branch7x7x3, branch_pool])
                                end_points['mixed_17x17x1280a'] = net
                        # mixed_9: 8 x 8 x 2048.
                        with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                            with CustomTFScope('mixed_8x8x2048a'):
                                with tf.device(devices[get_dev_id(1)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch1x1'):
                                        branch1x1 = ops.conv2d(net, 320, [1, 1])
                                with tf.device(devices[get_dev_id(2)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch3x3'):
                                        with CustomTFScope('conv_0'):
                                            branch3x3 = ops.conv2d(net, 384, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            b1 = ops.conv2d(branch3x3, 384, [1, 3])
                                        with CustomTFScope('conv_2'):
                                            b2 = ops.conv2d(branch3x3, 384, [3, 1])
                                        branch3x3 = tf.concat(axis=3, values=[b1, b2])
                                with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch3x3dbl'):
                                        with CustomTFScope('conv_0'):
                                            branch3x3dbl = ops.conv2d(net, 448, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch3x3dbl = ops.conv2d(branch3x3dbl, 384, [3, 3])
                                        with CustomTFScope('conv_2'):
                                            b1 = ops.conv2d(branch3x3dbl, 384, [1, 3])
                                        with CustomTFScope('conv_3'):
                                            b2 = ops.conv2d(branch3x3dbl, 384, [3, 1])
                                        branch3x3dbl = tf.concat(axis=3, values=[b1, b2])
                                with tf.device(devices[get_dev_id(3)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch_pool'):
                                        with CustomTFScope('avg_pool'):
                                            branch_pool = ops.avg_pool(net, [3, 3])
                                        with CustomTFScope('conv_0'):
                                            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                                net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3dbl, branch_pool])
                                end_points['mixed_8x8x2048a'] = net
                        # mixed_10: 8 x 8 x 2048.
                        with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                            with CustomTFScope('mixed_8x8x2048b'):
                                with tf.device(devices[get_dev_id(1)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch1x1'):
                                        branch1x1 = ops.conv2d(net, 320, [1, 1])

                                with tf.device(devices[get_dev_id(2)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch3x3'):
                                        with CustomTFScope('conv_0'):
                                            branch3x3 = ops.conv2d(net, 384, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            b1 = ops.conv2d(branch3x3, 384, [1, 3])
                                        with CustomTFScope('conv_2'):
                                            b2 = ops.conv2d(branch3x3, 384, [3, 1])
                                        branch3x3 = tf.concat(axis=3, values=[b1, b2])

                                with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch3x3dbl'):
                                        with CustomTFScope('conv_0'):
                                            branch3x3dbl = ops.conv2d(net, 448, [1, 1])
                                        with CustomTFScope('conv_1'):
                                            branch3x3dbl = ops.conv2d(branch3x3dbl, 384, [3, 3])
                                        with CustomTFScope('conv_2'):
                                            b1 = ops.conv2d(branch3x3dbl, 384, [1, 3])
                                        with CustomTFScope('conv_3'):
                                            b2 = ops.conv2d(branch3x3dbl, 384, [3, 1])
                                        branch3x3dbl = tf.concat(axis=3, values=[b1, b2])

                                with tf.device(devices[get_dev_id(3)]) if devices else ExitStack() as gs:
                                    with CustomTFScope('branch_pool'):
                                        with CustomTFScope('avg_pool'):
                                            branch_pool = ops.avg_pool(net, [3, 3])
                                        with CustomTFScope('conv_0'):
                                            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                                net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3dbl, branch_pool])
                                end_points['mixed_8x8x2048b'] = net
                        # Final pooling and prediction
                        with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                            with CustomTFScope('logits'):
                                shape = net.get_shape()
                                with CustomTFScope('avg_pool_drop_flatten'):
                                    net = ops.avg_pool(net, shape[1:3], padding='VALID', scope='pool')
                                    # 1 x 1 x 2048
                                    net = ops.dropout(net, dropout_keep_prob, scope='dropout')
                                    net = ops.flatten(net, scope='flatten')
                                    # 2048

                                with CustomTFScope('FC'):
                                    logits = ops.fc(net, num_classes, activation=None, scope='logits',
                                                restore=restore_logits)
                                    # 1000
                                    end_points['logits'] = logits
                                    end_points['predictions'] = tf.nn.softmax(logits, name='predictions')

                            if trainable:
                                predictions = NNModel.softmax_add_training_nodes(bs,
                                                                    end_points['logits'])
                            else:
                                predictions = end_points['predictions']

        return g, predictions


TENSORBOARD_LOG_DIR = './tb-logs/'
META_GRAPHS_DIR = './meta-graphs/'
N = 224
BS = 32
TRAINING=True

def get_handles(training):
    return InceptionV3().build(None, scope='inception_v3', bs=BS, trainable=training)

if __name__ == '__main__':
    g, trainer = InceptionV3().build(None, scope='inception_v3', bs=BS, trainable=TRAINING)
    with g.as_default():
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf.train.Saver().save(sess, META_GRAPHS_DIR + '/inception-v3')
        tf.train.export_meta_graph(filename=META_GRAPHS_DIR + 'inception-v3%s.meta' % ('-train' if TRAINING else ''))
        tf.summary.FileWriter('../meta-graphs/inception-v3%s'% ('-train' if TRAINING else ""), 
            graph = g)

    from tensorflow.python.client import timeline
    writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR + '/inception-v3%s/' % ('-train' if TRAINING else ''), g)
    with tf.Session(graph=g) as sess:
        tf.global_variables_initializer().run(session=sess)
        for i in range(10):
            print('.', end='')
            run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            sess.run([trainer],
                    run_metadata=run_metadata,
                    options=run_options)
            writer.add_run_metadata(run_metadata, "step-%d" % i)

        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        os.system('mkdir chrome-traces 2> /dev/null')
        chrome_trace_fname = './chrome-traces/inception_v3.trace'
        with open(chrome_trace_fname, 'w') as f:
                f.write(chrome_trace)

    writer.flush()
