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
    dataset = dataset.map(expand_dims)

    return dataset


def expand_dims(img, mask, dims):
    img = tf.expand_dims(img, axis=-1)
    mask = tf.expand_dims(mask, axis=-1)

    return img, mask, dims


def flip_img(axis, img, mask, dims):
    aux = tf.constant([axis])
    flipped_img = tf.reverse(img, axis=aux)
    flipped_mask = tf.reverse(mask, axis=aux)

    return flipped_img, flipped_mask, dims


def transpose_img(perm, img, mask, dims):
    transposed_img = tf.transpose(img, perm=perm)
    transposed_mask = tf.transpose(mask, perm=perm)

    # Shape is SIZE ^3, so no need for dims change
    # dims = [dims[perm[0]], dims[perm[1]], dims[perm[2]]]

    return transposed_img, transposed_mask, dims


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
    dataset_cc359_train = load_dataset(CC359_TRAIN)
    dataset_cc359_val = load_dataset(CC359_VAL)

    dataset_nfbs_train = load_dataset(NFBS_TRAIN)
    dataset_nfbs_val = load_dataset(NFBS_VAL)

    dataset_adni_train = load_dataset(ADNI_TRAIN)
    dataset_adni_val = load_dataset(ADNI_VAL)

    dataset_train = dataset_cc359_train.concatenate(dataset_nfbs_train)
    dataset_val = dataset_cc359_val.concatenate(dataset_nfbs_val)

    dataset_train = dataset_train.concatenate(dataset_adni_train)
    dataset_val = dataset_val.concatenate(dataset_adni_val)

    swapaxes = [[0, 2, 1, 3], [2, 0, 1, 3], [2, 1, 0, 3], [1, 0, 2, 3], [1, 2, 0, 3]]

    aux_train = [dataset_train]
    aux_val = [dataset_val]

    for each in swapaxes:
        aux_train.append(dataset_train.map(lambda i, m, d: transpose_img(each, i, m, d)))
        aux_val.append(dataset_val.map(lambda i, m, d: transpose_img(each, i, m, d)))

    # for d in range(len(aux_train)):
    #     aux = []
    #     add_all_flips(aux_train[d], [0, 1, 2], 0, aux)
    #     aux_train.extend(aux)

    # for d in range(len(aux_val)):
    #     aux = []
    #     add_all_flips(aux_val[d], [0, 1, 2], 0, aux)
    #     aux_val.extend(aux)

    for d in range(len(aux_train)):
        aux_train[d] = add_all_flips2(aux_train[d], [0, 1, 2], 0, aux_train[d])

    for d in range(len(aux_val)):
        aux_val[d] = add_all_flips2(aux_val[d], [0, 1, 2], 0, aux_val[d])

    for d in aux_train:
        dataset_train = dataset_train.concatenate(d)

    for d in aux_val:
        dataset_val = dataset_val.concatenate(d)

    return dataset_train, dataset_val
