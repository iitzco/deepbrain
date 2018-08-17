import tensorflow as tf
import numpy as np


def load_dataset(filename, size=None):
    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(decode)

        return dataset


if __name__ == "__main__":
    TF_RECORD = "train.tfrecord"
    dataset = load_dataset(TF_RECORD)
    iterator = dataset.make_one_shot_iterator()
    img, mask, dims = iterator.get_next()

    sess = tf.Session()
    while True:
        img_out, mask_out, dims_out = sess.run([img, mask, dims])

        print(dims_out)

