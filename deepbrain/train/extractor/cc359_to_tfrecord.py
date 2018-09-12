import tensorflow as tf
import os
import nibabel as nib
import numpy as np
import random
import re
from skimage.transform import resize


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


def parse_example(img_path, mask_path):
    img = nib.load(img_path)
    mask = nib.load(mask_path)

    assert img.shape == mask.shape

    dims = list(img.shape)

    img_np = img.get_fdata()
    mask_np = mask.get_fdata()

    img_np = resize(img_np, output_shape=(SIZE, SIZE, SIZE), mode='constant', anti_aliasing=True)
    mask_np = resize(mask_np, output_shape=(SIZE, SIZE, SIZE), mode='constant', anti_aliasing=True)

    img_np = (img_np / np.max(img_np)) * 255  # To [0-255]
    img_np = img_np.astype(np.uint8)

    boolean_mask = np.zeros_like(mask_np, dtype=np.uint8)
    boolean_mask[mask_np > 0.5] = 1

    tf_ex = tf.train.Example(features=tf.train.Features(feature={
            'dims': int64_list_feature(dims),
            'img': bytes_feature(img_np.reshape(-1).tobytes()),
            'mask': bytes_feature(boolean_mask.reshape(-1).tobytes()),
            }))

    return tf_ex


def process_dataset(p=0.9):
    img_dir = os.path.join(DATASET_DIR, "Original")
    mask_dir = os.path.join(DATASET_DIR, "STAPLE")

    train_writer = tf.python_io.TFRecordWriter(os.path.join(OUTPUT, "train.tfrecord"))
    val_writer = tf.python_io.TFRecordWriter(os.path.join(OUTPUT, "val.tfrecord"))

    regex = re.compile("[\w_]*.nii.gz")
    index = 0

    for f in os.listdir(img_dir):
        img_path = os.path.join(img_dir, f)

        if regex.match(f):
            name = f[:f.find('.')]
            mask_path = os.path.join(mask_dir, "{}_staple.nii.gz".format(name))

            tf_ex = parse_example(img_path, mask_path)

            if random.random() < p:
                train_writer.write(tf_ex.SerializeToString())
            else:
                val_writer.write(tf_ex.SerializeToString())

            index += 1

        if index % 10 == 0:
            print("Processed {} examples".format(index))

    train_writer.close()
    val_writer.close()

    print("Correctly created record for {} entries\n".format(index))


if __name__ == "__main__":
    process_dataset()
