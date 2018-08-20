import tensorflow as tf
from const import *

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

    return img, mask, dims


def normalize(img, mask, dims):
    img = tf.divide(tf.cast(img, tf.float32), 255)
    return img, mask, dims


def load_dataset(filename, size=None):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(decode)
    dataset = dataset.map(normalize)

    return dataset


def expand_dims(img, mask, dims):
    img = tf.expand_dims(img, axis=-1)
    mask = tf.expand_dims(mask, axis=-1)

    return img, mask, dims


def flip_img(axis, img, mask, dims):
    flipped_img = tf.reverse(img, axis=axis)
    flipped_mask = tf.reverse(mask, axis=axis)

    return flipped_img, flipped_mask, dims


def transpose_img(perm, img, mask, dims):
    transposed_img = tf.transpose(img, perm=perm)
    transposed_mask = tf.transpose(mask, perm=perm)

    dims = [dims[perm[0]], dims[perm[1]], dims[perm[2]]]

    return transposed_img, transposed_mask, dims


def load_all_datasets():

    dataset_cc359_train = load_dataset(CC359_TRAIN)
    dataset_cc359_val = load_dataset(CC359_VAL)

    dataset_nfbs_train = load_dataset(NFBS_TRAIN)
    dataset_nfbs_val = load_dataset(NFBS_VAL)

    dataset_train = dataset_cc359_train.concatenate(dataset_nfbs_train)
    dataset_val = dataset_cc359_val.concatenate(dataset_nfbs_val)

    swapaxes = [[0, 2, 1], [2, 0, 1], [2, 1, 0], [1, 0, 2], [1, 2, 0]]

    aux_train = [dataset_train]
    aux_val = [dataset_val]

    for each in swapaxes:
        aux_train.append(dataset_train.map(lambda i, m, d: transpose_img(each, i, m, d)))
        aux_val.append(dataset_val.map(lambda i, m, d: transpose_img(each, i, m, d)))

    for d in range(len(aux_train)):
        for j in [0, 1, 2]:
            aux_train.append(aux_train[d].map(lambda i, m, d: flip_img(j, i, m, d)))

    for d in range(len(aux_val)):
        for j in [0, 1, 2]:
            aux_val.append(aux_val[d].map(lambda i, m, d: flip_img(j, i, m, d)))

    for d in aux_train[1:]:
        dataset_train = dataset_train.concatenate(d)

    for d in aux_val[1:]:
        dataset_val = dataset_val.concatenate(d)

    return dataset_train, dataset_val
