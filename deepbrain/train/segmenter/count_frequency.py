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


def run():
    _dir = ADNI_DATASET_DIR

    labels = Path(os.path.join(_dir, "masks", "malpem"))
    
    ret = {}

    index = 0

    for each in os.listdir(labels):
        p = labels / each
        img = nib.load(str(p))

        x = img.get_fdata()
        x = x.astype(np.uint8)
        x = np.reshape(x, (-1))

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
