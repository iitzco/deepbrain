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


def parse_example(img_path, mask_path):
    img = nib.load(img_path)
    mask = nib.load(mask_path)

    dims = list(img.shape)

    img_np = img.get_fdata()
    mask_np = mask.get_fdata().squeeze()


    assert img_np.shape == mask_np.shape

    img_np = resize(img_np, output_shape=(SIZE, SIZE, SIZE), mode='constant', anti_aliasing=True)
    mask_np = resize(mask_np, output_shape=(SIZE, SIZE, SIZE), mode='constant', anti_aliasing=True)

    if np.isnan(np.max(img_np)):
        raise ValueError()

    if np.isinf(np.max(img_np)):
        raise ValueError()

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


def process_dataset(prob=0.9):
    _dir = ADNI_DATASET_DIR

    brain_masks = Path(os.path.join(_dir, "brain_masks"))
    originals = list(Path(os.path.join(_dir, "ADNI")).glob("**/*.nii"))

    train_writer = tf.python_io.TFRecordWriter(ADNI_TRAIN)
    val_writer = tf.python_io.TFRecordWriter(ADNI_VAL)

    index = 0
    for each in os.listdir(brain_masks):
        e = Path(each)

        for orig in originals:
            if e.stem == orig.name:

                img_path = str(orig)
                mask_path = str(brain_masks / each)

                try:
                    tf_ex = parse_example(img_path, mask_path)
                    index += 1
                except OSError:
                    # print("OSError")
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
