# Count class frequency to deal with unbalance
import tensorflow as tf
import os
import nibabel as nib
import numpy as np
import random
import re
from skimage.transform import resize
from pathlib import Path

from const import *

# CSF: 1, 2, 23, 24, 0, 18 -> 1
# WM: 16, 17 -> 2
# GM: Rest -> 3
# Brain Stem: 7 -> 4
# Cerebellum WM: 12, 13 -> 5
# Cerebellum GM: 10, 11, 36, 37, 38 -> 6

def shrink_labels(labels):
    labels[np.isin(labels, [1,2,23,24,0,18])] = 1
    labels[np.isin(labels, [16,17])] = 2
    labels[~np.isin(labels, [1,2,23,24,0,18,16,17,7,12,13,10,11,36,37,38])] = 3
    labels[np.isin(labels, [7])] = 4
    labels[np.isin(labels, [12,13])] = 5
    labels[np.isin(labels, [10,11,36,37,38])] = 6
    return labels


def run():
    _dir = ADNI_DATASET_DIR

    labels = Path(os.path.join(_dir, "masks", "malpem"))
    brains = Path(os.path.join(_dir, "masks", "brain_masks"))
    
    ret = {}

    index = 0

    for each in os.listdir(labels):
        aux = each[7:]

        p = labels / each
        b = brains / aux

        img = nib.load(str(p))
        brain = (nib.load(str(b)).get_fdata().squeeze()) == 1

        x = img.get_fdata()
        x = x.astype(np.uint8).squeeze()

        assert x.shape == brain.shape

        x = x[brain]
        x = shrink_labels(x)

        y = np.bincount(x)
        ii = np.nonzero(y)[0]

        index +=1

        if index % 100 == 0:
            print("Processed {}".format(index))

        for k, v in zip(ii,y[ii]):
                ret[k] = ret.get(k, 0) + v

    print(ret)


if __name__ == "__main__":
    run()
