import tensorflow as tf
import os
import nibabel as nib
import numpy as np
import random
import re
from skimage.transform import resize
from pathlib import Path


from const import *


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def parse_example(img_path, brain_path, labels_path):
    img = nib.load(img_path)
    labels = nib.load(labels_path)
    brain = nib.load(brain_path)

    dims = list(img.shape)

    img_np = img.get_fdata()
    labels_np = labels.get_fdata().squeeze()
    brain_np = brain.get_fdata().squeeze()

    assert img_np.shape == labels_np.shape

    img_np = resize(img_np, output_shape=(SIZE, SIZE, SIZE), mode='constant', anti_aliasing=True)
    labels_np = resize(labels_np, output_shape=(SIZE, SIZE, SIZE), mode='constant', anti_aliasing=True)
    brain_np = resize(brain_np, output_shape=(SIZE, SIZE, SIZE), mode='constant', anti_aliasing=True)

    if np.isnan(np.max(img_np)):
        raise ValueError()

    if np.isinf(np.max(img_np)):
        raise ValueError()

    img_np = (img_np / np.max(img_np)) * 255  # To [0-255]

    img_np = img_np.astype(np.uint8)
    img_np[brain_np <= 0.5] = 0 # Strip brain

    labels_np = labels_np.astype(np.uint8)

    tf_ex = tf.train.Example(features=tf.train.Features(feature={
            'dims': int64_list_feature(dims),
            'img': bytes_feature(img_np.reshape(-1).tobytes()),
            'labels': bytes_feature(labels_np.reshape(-1).tobytes()),
            }))

    return tf_ex


def process_dataset(prob=0.9):
    _dir = ADNI_DATASET_DIR

    labels_masks = Path(os.path.join(_dir, "masks", "malpem"))
    brain_masks = Path(os.path.join(_dir, "masks", "brain_masks"))

    originals = list(Path(os.path.join(_dir, "ADNI")).glob("**/*.nii"))

    train_writer = tf.python_io.TFRecordWriter(ADNI_TRAIN)
    val_writer = tf.python_io.TFRecordWriter(ADNI_VAL)

    index = 0
    for each in os.listdir(labels_masks):
        e = Path(each)
        aux = str(e)[7:] # Remove the MALPEM- prefix

        for orig in originals:
            if aux[:-3] == orig.name:

                img_path = str(orig)
                labels_path = str(labels_masks / each)
                brain_path = str(brain_masks / aux)

                try:
                    tf_ex = parse_example(img_path, brain_path, labels_path)
                    index += 1
                except OSError:
                    print("OSError")
                    break
                except ValueError:
                    print("ValueError")
                    break

                if random.random() < prob:
                    train_writer.write(tf_ex.SerializeToString())
                else:
                    val_writer.write(tf_ex.SerializeToString())

                if index % 10 == 0:
                    print("Processed {} examples".format(index))
                break

    train_writer.close()
    val_writer.close()

    print("Correctly created record for {} entries\n".format(index))


if __name__ == "__main__":
    process_dataset()
