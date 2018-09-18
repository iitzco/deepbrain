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
