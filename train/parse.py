
import tensorflow as tf
import numpy as np
from bitstring import ConstBitStream
import random
import multiprocessing as mp
import sys
import os
import time
from tfprocess import TFProcess
from archive import GameArchive

from archive import BOARD_SIZE, BOARD_SQ, HISTORY_STEP, INPUT_CHANNELS

# Sane values are from 4096 to 64 or so. The maximum depends on the amount
# of RAM in your GPU and the network size. You need to adjust the learning rate
# if you change this.
BATCH_SIZE = 64



def weight_variable(name, shape):
    """Xavier initialization"""
    stddev = np.sqrt(2.0 / (sum(shape)))
    initial = tf.truncated_normal(shape, stddev=stddev)
    weights = tf.Variable(initial, name=name)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
    return weights

# Bias weights for layers not followed by BatchNorm
# We do not regularlize biases, so they are not
# added to the regularlizer collection
def bias_variable(name, shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name=name)


def batch_norm(x, training, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x,
                        decay=momentum, 
                        updates_collections=None,
                        epsilon=epsilon,
                        scale=True,
                        is_training=training,
                        scope=name)

def conv2d(x, filters, kernel, name="conv"):
    with tf.variable_scope(name):
        w = weight_variable('W', [kernel, kernel, x.get_shape().as_list()[1], filters])
        conv = tf.nn.conv2d(x, w, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')
        return conv

def fc(x, output_size, name="fc"):
    with tf.variable_scope(name):
        matrix = weight_variable("W", [x.get_shape().as_list()[1], output_size])
        bias = bias_variable("b", [output_size])
        return tf.add(tf.matmul(x, matrix), bias)

def bn_conv(x, filters, kernel, training, name):
    return batch_norm(conv2d(x, filters, kernel, name=name), training=training, name=name)


def training_model(x, filters, res_blocks, training):

    h = tf.nn.relu(bn_conv(x, filters, 3, training, "conv0"))
    for i in range(1, res_blocks+1):
        orig = tf.identity(h)
        h = tf.nn.relu(bn_conv(h, filters, 3, training, "conv%d/0" % (i,)))
        h = bn_conv(h, filters, 3, training, "conv%d/1" % (i,))
        h = tf.nn.relu(tf.add(h, orig))

    with tf.variable_scope("policy_head"):
        ph = tf.nn.relu(bn_conv(h, 2, 1, training, "conv"))
        ph = tf.reshape(ph, [-1, 2*BOARD_SQ])
        ph = fc(ph, BOARD_SQ+1, "fc")

    with tf.variable_scope("value_head"):
        vh = tf.nn.relu(bn_conv(h, 1, 1, training, "conv"))
        vh = tf.reshape(vh, [-1, BOARD_SQ])
        vh = tf.nn.relu(fc(vh, 256, "fc1"))
        vh = tf.nn.tanh(fc(vh, 1, "fc2"))

    return ph, vh



def remap_vertex(vertex, symmetry):
    """
        Remap a go board coordinate according to a symmetry.
    """
    assert vertex >= 0 and vertex < BOARD_SQ
    x = vertex % BOARD_SIZE
    y = vertex // BOARD_SIZE
    if symmetry >= 4:
        x, y = y, x
        symmetry -= 4
    if symmetry == 1 or symmetry == 3:
        x = BOARD_SIZE - x - 1
    if symmetry == 2 or symmetry == 3:
        y = BOARD_SIZE - y - 1
    return y * BOARD_SIZE + x

class ChunkParser(object):
    def __init__(self, chunks, workers=None):
        # Build probility reflection tables. The last element is 'pass' and is identity mapped.
        self.prob_reflection_table = [[remap_vertex(vertex, sym) for vertex in range(BOARD_SQ)]+[BOARD_SQ] for sym in range(8)]
        # Build full 16-plane reflection tables.
        self.full_reflection_table = [
            [remap_vertex(vertex, sym) + p * BOARD_SQ for p in range(2*HISTORY_STEP) for vertex in range(BOARD_SQ) ]
                for sym in range(8) ]
        # Convert both to np.array. This avoids a conversion step when they're actually used.
        self.prob_reflection_table = [ np.array(x, dtype=np.int64) for x in self.prob_reflection_table ]
        self.full_reflection_table = [ np.array(x, dtype=np.int64) for x in self.full_reflection_table ]
        # Build the all-zeros and all-ones flat planes, used for color-to-move.
        self.flat_planes = [ b'\0' * BOARD_SQ, b'\1' * BOARD_SQ ]

        # Start worker processes, leave 2 for TensorFlow
        if workers is None:
            workers = max(1, mp.cpu_count() - 2)
        print("Using {} worker processes.".format(workers))
        self.readers = []
        for _ in range(workers):
            read, write = mp.Pipe(False)
            mp.Process(target=self.task,
                       args=(chunks, write)).start()
            self.readers.append(read)


    def convert_train_data(self, game, index, symmetry):
        """
            Convert textual training data to a tf.train.Example

            Converts a set of 19 lines of text into a pythonic dataformat.
            [[plane_1],[plane_2],...],...
            [probabilities],...
            winner,...
        """
        assert symmetry >= 0 and symmetry < 8

        planes, probabilities, player_is_black, result = game.encode(index)

        # We flatten to a single array of len 16*19*19, type=np.uint8
        planes = np.concatenate(planes)

        # We use the full length reflection tables to apply symmetry
        # to all 16 planes simultaneously
        planes = planes[self.full_reflection_table[symmetry]]
        # Convert the array to a byte string
        planes = [ planes.tobytes() ]

        # Now we add the two final planes, being the 'color to move' planes.
        # These already a fully symmetric, so we add them directly as byte
        # strings of length 361.
        if player_is_black:
            stm = 0
        else:
            stm = 1

        planes.append(self.flat_planes[1 - stm])
        planes.append(self.flat_planes[stm])

        # Flatten all planes to a single byte string
        planes = b''.join(planes)
        assert len(planes) == (INPUT_CHANNELS * BOARD_SQ)

        # Apply symmetries to the probabilities.
        probabilities = probabilities[self.prob_reflection_table[symmetry]]
        assert len(probabilities) == (BOARD_SQ+1)

        # Load the game winner color.
        winner = float(result)

        # Construct the Example protobuf
        example = tf.train.Example(features=tf.train.Features(feature={
            'planes' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[planes])),
            'probs' : tf.train.Feature(float_list=tf.train.FloatList(value=probabilities)),
            'winner' : tf.train.Feature(float_list=tf.train.FloatList(value=[winner]))}))
        return example.SerializeToString()


    def task(self, chunks, writer):
        while True:
            random.shuffle(chunks)
            for chunk in chunks:
                chunk.parse()
                item_count = len(chunk.valid_steps)
                # Pick only 1 in every 16 positions
                picked_items = random.sample(range(item_count),
                                                 (item_count + 15) // 16)

                for item_idx in picked_items:
                    item = chunk.valid_steps[item_idx]
                    # Pick a random symmetry to apply
                    symmetry = random.randrange(8)
                    data = self.convert_train_data(chunk, item, symmetry)
                    # Send it down the pipe.
                    writer.send_bytes(data)


    def parse_chunk(self):
        while True:
            for r in self.readers:
                yield r.recv_bytes()

# Convert a tf.train.Example protobuf into a tuple of tensors
# NB: This conversion is done in the tensorflow graph, NOT in python.
def _parse_function(example_proto):
    features = {"planes": tf.FixedLenFeature((1), tf.string),
                "probs": tf.FixedLenFeature((BOARD_SQ+1), tf.float32),
                "winner": tf.FixedLenFeature((1), tf.float32)}
    parsed_features = tf.parse_single_example(example_proto, features)
    # We receives the planes as a byte array, but we really want
    # floats of shape (18, 19*19), so decode, cast, and reshape.
    planes = tf.decode_raw(parsed_features["planes"], tf.uint8)
    planes = tf.to_float(planes)
    planes = tf.reshape(planes, (INPUT_CHANNELS, BOARD_SQ))
    # the other features are already in the correct shape as return as-is.
    return planes, parsed_features["probs"], parsed_features["winner"]


def main(args):

    trainning = GameArchive('../../data/train1.data')
    train_parser = ChunkParser(trainning.games)
    dataset = tf.data.Dataset.from_generator(
        train_parser.parse_chunk, output_types=(tf.string))
    dataset = dataset.shuffle(1 << 18)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(4)
    train_iterator = dataset.make_one_shot_iterator()

    testing = GameArchive('../../data/test1.data')
    test_parser = ChunkParser(testing.games)
    dataset = tf.data.Dataset.from_generator(
        test_parser.parse_chunk, output_types=(tf.string))
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(4)
    test_iterator = dataset.make_one_shot_iterator()

    tfprocess = TFProcess()
    tfprocess.init(dataset, train_iterator, test_iterator)

    if args:
        restore_file = args.pop(0)
        tfprocess.restore(restore_file)
    while True:
        tfprocess.process(BATCH_SIZE)

if __name__ == "__main__":
    main(sys.argv[1:])
    mp.freeze_support()
