import os
import nibabel as nib
import numpy as np
import random
import re
from skimage.transform import resize
from pathlib import Path


from const import *

def process_dataset(prob=0.9):
    _dir = ADNI_DATASET_DIR

    brain_masks = Path(os.path.join(_dir, "masks", "brain_masks"))
    originals = list(Path(os.path.join(_dir, "ADNI")).glob("**/*.nii"))

    index = 0
    for each in os.listdir(brain_masks):
        e = Path(each)

        for orig in originals:
            if e.stem == orig.name:
                index+=1
                if index % 10 == 0:
                    print(index)
                break


    print("Found {}".format(index))


if __name__ == "__main__":
    process_dataset()
