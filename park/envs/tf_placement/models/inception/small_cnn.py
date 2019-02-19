from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops

import ops
import scopes
import sys
import random
from nn_model import *

class SmallCNN(NNModel):

    def build(self, inputs=None,
                    trainable=True,
                    bs=128,
                    num_filters=32,
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

        def get_dev_id(i):
            N = len(devices)
            if device_placement == 'random':
                dev_id = random.randint(0, N-1)
            elif device_placement == 'alternate':
                dev_id = i% N
            else:
                # device_placement == 'expert' or otherwise
                dev_id = 0
            return dev_id

        print('BS: '+ str(bs))
        import pdb; pdb.set_trace()
        if inputs is None:
            inputs = tf.ones((bs, 299, 299, 3))

        with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                            is_training=trainable):
            with tf.device(devices[get_dev_id(0)]) if devices else ExitStack() as gs:
                with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                                        stride=1, padding='VALID'):
                    # 299 x 299 x 3
                    ret = ops.conv2d(inputs, num_filters, [3, 3], stride=2, scope='conv0')

        return tf.get_default_graph(), ret
