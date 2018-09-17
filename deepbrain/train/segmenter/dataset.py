import tensorflow as tf
from const import *

def decode(serialized_example):
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'dims': tf.FixedLenFeature([3], tf.int64),
                                           'img': tf.FixedLenFeature([], tf.string),
                                           'labels': tf.FixedLenFeature([], tf.string),
                                       })

    dims = features['dims']

    img = tf.decode_raw(features['img'], tf.uint8)
    labels = tf.decode_raw(features['labels'], tf.uint8)

    aux = tf.constant([SIZE, SIZE, SIZE])
    img = tf.reshape(img, shape=aux)
    labels = tf.reshape(labels, shape=aux)

    return img, labels, dims


def normalize(img, labels, dims):
    img = tf.divide(tf.cast(img, tf.float32), 255)
    return img, labels, dims


def load_dataset(filename, size=None):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(decode)
    dataset = dataset.map(normalize)
    dataset = dataset.map(to_one_hot)
    dataset = dataset.map(expand_dims)

    return dataset


def expand_dims(img, labels, dims):
    img = tf.expand_dims(img, axis=-1)

    return img, labels, dims


def to_one_hot(img, labels, dims):
    aux = tf.reshape(labels, [-1])
    one_hot = tf.one_hot(aux, LABELS)
    one_hot_labels = tf.reshape(one_hot, [SIZE, SIZE, SIZE, -1])

    return img, one_hot_labels, dims


def flip_img(axis, img, labels, dims):
    aux = tf.constant([axis])
    flipped_img = tf.reverse(img, axis=aux)
    flipped_mask = tf.reverse(labels, axis=aux)

    return flipped_img, flipped_mask, dims


def transpose_img(perm, img, labels, dims):
    transposed_img = tf.transpose(img, perm=perm)
    transposed_labels = tf.transpose(labels, perm=perm)

    # Shape is SIZE ^3, so no need for dims change
    # dims = [dims[perm[0]], dims[perm[1]], dims[perm[2]]]

    return transposed_img, transposed_labels, dims


# All flips is 2^n combination, being n the axis. For each axis, perform flip or not.
# That's how you get (L-R, T-B, P-A), (L-R, T-B, P-A), (L-R, B-T, P-A)... (left, right, top, bottom, posterior, anterior)
def add_all_flips(dataset, axes, index, list_to_fill):
    if index == len(axes):
        list_to_fill.append(dataset)
        return

    # Call without flip
    add_all_flips(dataset, axes, index+1, list_to_fill)

    # Call with flip
    flipped_dataset = dataset.map(lambda i, m, d: flip_img(axes[index], i, m, d))
    add_all_flips(flipped_dataset, axes, index+1, list_to_fill)


# Change implementation because of performance issues.
def add_all_flips2(dataset, axes, index, final_dataset):
    if index == len(axes):
        return final_dataset.concatenate(dataset)

    # Call without flip
    final_dataset = add_all_flips2(dataset, axes, index+1, final_dataset)

    # Call with flip
    flipped_dataset = dataset.map(lambda i, m, d: flip_img(axes[index], i, m, d))
    return add_all_flips2(flipped_dataset, axes, index+1, final_dataset)


def load_all_datasets():
    dataset_train = load_dataset(ADNI_TRAIN)
    dataset_val = load_dataset(ADNI_VAL)

    swapaxes = [[0, 2, 1, 3], [2, 0, 1, 3], [2, 1, 0, 3], [1, 0, 2, 3], [1, 2, 0, 3]]

    aux_train = [dataset_train]
    aux_val = [dataset_val]

    for each in swapaxes:
        aux_train.append(dataset_train.map(lambda i, m, d: transpose_img(each, i, m, d)))
        aux_val.append(dataset_val.map(lambda i, m, d: transpose_img(each, i, m, d)))

    for d in range(len(aux_train)):
        aux_train[d] = add_all_flips2(aux_train[d], [0, 1, 2], 0, aux_train[d])

    for d in range(len(aux_val)):
        aux_val[d] = add_all_flips2(aux_val[d], [0, 1, 2], 0, aux_val[d])

    for d in aux_train:
        dataset_train = dataset_train.concatenate(d)

    for d in aux_val:
        dataset_val = dataset_val.concatenate(d)

    return dataset_train, dataset_val
