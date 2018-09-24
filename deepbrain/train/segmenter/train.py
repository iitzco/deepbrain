import tensorflow as tf
import subprocess
import sys
import numpy as np
from halo import Halo

from const import SIZE, LABELS, FILTERED_LABELS, LABEL_MAP, FREQ_LIST, FREQ_PROP
from dataset import load_all_datasets


def model(img, labels, dims):

    init = tf.contrib.layers.xavier_initializer()

    training = tf.placeholder_with_default(True, shape=[], name="training")

    input_ = tf.placeholder_with_default(img, shape=[None, SIZE, SIZE, SIZE, 1], name="img")
    dims = tf.placeholder_with_default(dims, shape=[None, 3], name="dim")

    out = tf.cast(input_, dtype=tf.float32)
    
    out = tf.layers.conv3d(out, filters=8, kernel_size=3, activation=tf.nn.relu, kernel_initializer=init, padding="same")
    out = tf.layers.conv3d(out, filters=8, kernel_size=3, activation=tf.nn.relu, kernel_initializer=init, padding="same")
    out = tf.layers.batch_normalization(out, training=training)

    conv1 = out

    out = tf.layers.max_pooling3d(out, pool_size=2, strides=2)

    out = tf.layers.dropout(out, rate=0.4, training=training)

    out = tf.layers.conv3d(out, filters=16, kernel_size=5, activation=tf.nn.relu, kernel_initializer=init, padding="same")
    out = tf.layers.conv3d(out, filters=16, kernel_size=5, activation=tf.nn.relu, kernel_initializer=init, padding="same")
    out = tf.layers.batch_normalization(out, training=training)

    conv2 = out

    out = tf.layers.max_pooling3d(out, pool_size=2, strides=2)

    out = tf.layers.dropout(out, rate=0.4, training=training)

    out = tf.layers.conv3d(out, filters=32, kernel_size=7, activation=tf.nn.relu, kernel_initializer=init, padding="same")
    out = tf.layers.conv3d(out, filters=32, kernel_size=7, activation=tf.nn.relu, kernel_initializer=init, padding="same")
    out = tf.layers.batch_normalization(out, training=training)

    # conv3 = out

    # out = tf.layers.max_pooling3d(out, pool_size=2, strides=2)

    # out = tf.layers.dropout(out, rate=0.3, training=training)

    # out = tf.layers.conv3d_transpose(out, filters=128, kernel_size=5, strides=2, kernel_initializer=init, padding="same", use_bias=False)
    # out = tf.concat((out, conv3), axis=-1)
    # out = tf.layers.conv3d(out, filters=128, kernel_size=5, activation=tf.nn.relu, kernel_initializer=init, padding="same")
    # out = tf.layers.conv3d(out, filters=128, kernel_size=5, activation=tf.nn.relu, kernel_initializer=init, padding="same")

    # out = tf.layers.dropout(out, rate=0.3, training=training)

    out = tf.layers.conv3d_transpose(out, filters=16, kernel_size=5, strides=2, kernel_initializer=init, padding="same", use_bias=False)
    out = tf.concat((out, conv2), axis=-1)
    out = tf.layers.conv3d(out, filters=16, kernel_size=5, activation=tf.nn.relu, kernel_initializer=init, padding="same")
    out = tf.layers.batch_normalization(out, training=training)

    out = tf.layers.dropout(out, rate=0.4, training=training)

    out = tf.layers.conv3d_transpose(out, filters=8, kernel_size=3, strides=2, kernel_initializer=init, padding="same", use_bias=False)
    out = tf.concat((out, conv1), axis=-1)
    out = tf.layers.conv3d(out, filters=8, kernel_size=3, activation=tf.nn.relu, kernel_initializer=init, padding="same")
    out = tf.layers.batch_normalization(out, training=training)

    out = tf.layers.dropout(out, rate=0.4, training=training)

    out = tf.layers.conv3d(out, filters=FILTERED_LABELS, kernel_size=1, kernel_initializer=init, padding="same")

    softmax_out = tf.nn.softmax(out, name="softmax")

    labels = tf.subtract(labels, tf.constant(1, dtype=tf.uint8))

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(tf.squeeze(labels, axis=-1), tf.int32), logits=out)

    # Ignore background
    brain_mask = tf.squeeze(tf.greater(img, 0), axis=-1)

    freq = tf.constant(FREQ_PROP, dtype=tf.float32)
    weights = 1 / freq
    
    loss = tf.boolean_mask(loss, brain_mask)
    labels2 = tf.boolean_mask(tf.squeeze(labels), brain_mask)

    w = tf.gather(weights, tf.cast(labels2, tf.int32))

    loss = tf.multiply(w, loss)
    loss = tf.reduce_mean(loss)

    pred = tf.cast(tf.argmax(out, axis=-1, name="pred"), tf.uint8)
    pred = tf.boolean_mask(pred, brain_mask)
    correct_pred = tf.equal(pred, labels2)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    for k, v in LABEL_MAP.items():
        correct = tf.equal(pred, tf.constant(v, dtype=np.uint8))
        gt = tf.equal(labels2, tf.constant(v, dtype=np.uint8))
        intersection = tf.reduce_sum(tf.cast(tf.logical_and(correct, gt), dtype=tf.float32))
        union = tf.reduce_sum(tf.cast(tf.logical_or(correct, gt), dtype=tf.float32))
        tf.summary.scalar("iou_{}".format(k), intersection / union)
    
    tf.summary.scalar("acc", accuracy)
    tf.summary.scalar("loss", loss)
    
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.96, staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        upd = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
        
    merged = tf.summary.merge_all()
    
    return training, img, labels, out, merged, upd


def load_iterators(train_dataset, val_dataset):
    batch_size = 1

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

    train_dataset, val_dataset = load_all_datasets()

    handle, training_iterator, validation_iterator, next_element = load_iterators(train_dataset, val_dataset)

    training, img, labels, out, merged, upd = model(*next_element)

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
    # img_out, labels_out = sess.run([img, labels], feed_dict={handle: training_handle})
    # img_out = img_out[0].squeeze()
    # labels_out = labels_out[0].squeeze()

    # import nibabel as nib
    # nib.save(nib.Nifti1Image(img_out, np.eye(4)), "original.nii")
    # nib.save(nib.Nifti1Image(labels_out, np.eye(4)), "seg.nii")
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

