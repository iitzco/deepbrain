import tensorflow as tf
import subprocess
import sys
import numpy as np

from halo import Halo

DATASET_TRAIN = "./train.tfrecord"
DATASET_VAL = "./val.tfrecord"

from const import SIZE

def decode(serialized_example):
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'dims': tf.FixedLenFeature([3], tf.int64),
                                           'img': tf.FixedLenFeature([], tf.string),
                                           'mask': tf.FixedLenFeature([], tf.string),
                                       })

    dims = features['dims']

    img = tf.decode_raw(features['img'], tf.uint8)
    mask = tf.decode_raw(features['mask'], tf.uint8)

    aux = tf.constant([SIZE, SIZE, SIZE])
    img = tf.reshape(img, shape=aux)
    mask = tf.reshape(mask, shape=aux)

    img = tf.expand_dims(img, axis=-1)
    mask = tf.expand_dims(mask, axis=-1)

    return img, mask, dims


def normalize(img, mask, dims):
    img = tf.divide(tf.cast(img, tf.float32), 255)
    return img, mask, dims


def load_dataset(filename, size=None):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(decode)
    dataset = dataset.map(normalize)
    
    return dataset


def model(img, mask, dims):

    init = tf.contrib.layers.xavier_initializer()

    training = tf.placeholder_with_default(True, shape=[], name="training")

    input_ = tf.placeholder_with_default(img, shape=[None, SIZE, SIZE, SIZE, 1], name="img")
    dims = tf.placeholder_with_default(dims, shape=[None, 3], name="dim")

    out = tf.cast(input_, dtype=tf.float32)
    
    out = tf.layers.conv3d(out, filters=8, kernel_size=3, activation=tf.nn.relu, kernel_initializer=init, padding="same")
    out = tf.layers.max_pooling3d(out, pool_size=2, strides=2)

    out = tf.layers.dropout(out, rate=0.3, training=training)

    out = tf.layers.conv3d(out, filters=16, kernel_size=3, activation=tf.nn.relu, kernel_initializer=init, padding="same")
    out = tf.layers.max_pooling3d(out, pool_size=2, strides=2)

    out = tf.layers.dropout(out, rate=0.3, training=training)

    out = tf.layers.conv3d(out, filters=32, kernel_size=3, activation=tf.nn.relu, kernel_initializer=init, padding="same")
    out = tf.layers.max_pooling3d(out, pool_size=2, strides=2)

    out = tf.layers.dropout(out, rate=0.3, training=training)

    out = tf.layers.conv3d_transpose(out, filters=32, kernel_size=3, strides=2, kernel_initializer=init, padding="same", use_bias=False)
    out = tf.layers.dropout(out, rate=0.3, training=training)

    out = tf.layers.conv3d_transpose(out, filters=16, kernel_size=3, strides=2, kernel_initializer=init, padding="same", use_bias=False)
    out = tf.layers.dropout(out, rate=0.3, training=training)

    out = tf.layers.conv3d_transpose(out, filters=8, kernel_size=3, strides=2, kernel_initializer=init, padding="same", use_bias=False)
    out = tf.layers.dropout(out, rate=0.3, training=training)

    out = tf.layers.conv3d(out, filters=1, kernel_size=1, kernel_initializer=init, padding="same")

    sigm_out = tf.nn.sigmoid(out, name="prob")

    threshold = tf.multiply(tf.ones_like(mask, dtype=tf.float32), 0.5)
    pred = tf.greater(sigm_out, threshold, "pred")
    mask_bool = tf.greater(tf.cast(mask, dtype=tf.float32), threshold)

    _and = tf.logical_and(pred, mask_bool)
    _or = tf.logical_or(pred, mask_bool)

    _and = tf.reduce_sum(tf.cast(_and, tf.float32), axis=[1, 2, 3, 4])
    _or = tf.reduce_sum(tf.cast(_or, tf.float32), axis=[1, 2, 3, 4])

    iou = tf.reduce_mean(_and / _or)
    tf.summary.scalar("iou", iou)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(mask, tf.float32), logits=out)
    loss = tf.reduce_mean(loss)

    tf.summary.scalar("loss", loss)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        upd = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        
    merged = tf.summary.merge_all()
    
    return training, img, mask, out, merged, upd


def load_iterators():
    batch_size = 2

    train_dataset = load_dataset(DATASET_TRAIN)
    val_dataset = load_dataset(DATASET_VAL)

    train_dataset = train_dataset.shuffle(batch_size)
    train_dataset = train_dataset.repeat()

    val_dataset = val_dataset.shuffle(batch_size)
    val_dataset = val_dataset.repeat()

    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=batch_size)

    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(buffer_size=batch_size)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

    next_element = iterator.get_next()

    training_iterator = train_dataset.make_one_shot_iterator()
    validation_iterator = val_dataset.make_one_shot_iterator()

    return handle, training_iterator, validation_iterator, next_element


def run():
    tf.reset_default_graph()

    handle, training_iterator, validation_iterator, next_element = load_iterators()

    training, img, mask, out, merged, upd = model(*next_element)

    saver = tf.train.Saver(max_to_keep=2)

    sess = tf.Session()

    train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
    val_writer = tf.summary.FileWriter('./logs/val', sess.graph)

    sess.run(tf.global_variables_initializer())

    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    i = 0

    spinner = Halo(text='Training', spinner='dots')
    subprocess.Popen(["tensorboard", "--logdir", "./logs", "--port", "6006", "--host", "0.0.0.0"])

    spinner.start()
    while True:
        m, _ = sess.run([merged, upd], feed_dict={handle: training_handle})
        train_writer.add_summary(m, i)

        if i % 100 == 0:
            m = sess.run(merged, feed_dict={training: False, handle: validation_handle})
            val_writer.add_summary(m, i)
            val_writer.flush()

        if i % 1000 == 0 and i > 0:
            # Save model
            saver.save(sess, "./models/model.ckpt", global_step=i)

        i+=1

    spinner.stop()

if __name__ == "__main__":
    run()
