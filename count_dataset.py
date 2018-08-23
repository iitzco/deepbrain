import tensorflow as tf
import subprocess
import sys
import numpy as np
from halo import Halo

from const import SIZE
from dataset import load_all_datasets

BATCH_SIZE = 1

def load_iterators(train_dataset, val_dataset):
    batch_size = BATCH_SIZE

    train_dataset = train_dataset.shuffle(batch_size)

    val_dataset = val_dataset.shuffle(batch_size)

    train_dataset = train_dataset.batch(batch_size)

    val_dataset = val_dataset.batch(batch_size)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

    next_element = iterator.get_next()

    training_iterator = train_dataset.make_one_shot_iterator()
    validation_iterator = val_dataset.make_one_shot_iterator()

    return handle, training_iterator, validation_iterator, next_element


def run():
    tf.reset_default_graph()

    train_dataset, val_dataset = load_all_datasets()

    handle, training_iterator, validation_iterator, next_element = load_iterators(train_dataset, val_dataset)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    i = 0

    while True:
        try:
            sess.run(next_element[0], feed_dict={handle: training_handle})
            i+=BATCH_SIZE
        except:
            print("Train examples: {} +/- {}".format(i, BATCH_SIZE))
            break

    i = 0

    while True:
        try:
            sess.run(next_element[0], feed_dict={handle: validation_handle})
            i+=BATCH_SIZE
        except:
            print("Validation examples: {} +/- {}".format(i, BATCH_SIZE))
            break

if __name__ == "__main__":
    run()

