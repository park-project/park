import tensorflow as tf
import argparse
import sys
sys.path.append('../')
from custom_tf_scope import CustomTFScope
from tensorflow.python.framework import dtypes
from contextlib import ExitStack

class NNModel(object):
    def __init__(self):
        self.dtype = dtypes.float32
        return

    @staticmethod
    def softmax_add_training_nodes(bs, y_pred):
        with tf.name_scope('train_ops'):
            with tf.name_scope('dummy_labels'):
                labels = tf.constant(0, shape=[bs], dtype=tf.int32)
            with tf.name_scope('cross_entropy_loss'):
                cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                                        labels=labels, logits=y_pred)
                print(labels.shape.as_list())
                print(y_pred.shape.as_list())
            with tf.name_scope('optimizer'):
                trainer = tf.train.AdamOptimizer(learning_rate=1e-3).\
                        minimize(cross_entropy, colocate_gradients_with_ops=True)

        return trainer


TENSORBOARD_LOG_DIR = 'tb-logs/'
META_GRAPHS_DIR = 'meta-graphs/'
N = 224
BS = 128
TRAINING=False

def get_handles(s):
    if 'inception' in s:
        from inception.inception_model import get_handles
    elif 'fnn' in s:
        from fnn import get_handles
    return get_handles(TRAINING)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--meta-graph', action='store_true', dest='meta_graph')
    args = parser.parse_args()

    g, trainer = get_handles(args.model)
    if args.meta_graph:
        with g.as_default():
            tf.train.export_meta_graph(filename=META_GRAPHS_DIR + '%s%s.meta' \
                                    % (args.model, '-train' if TRAINING else ''))
        sys.exit(0)

    writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR + '%s%s/' \
                            % (args.model, '-train' if TRAINING else ''), g)

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
    writer.flush()

