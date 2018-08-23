import sys
import nibabel as nib
import numpy as np
from skimage.transform import resize

from const import *

img = nib.load(sys.argv[1])
affine = img.affine
img = img.get_fdata()

shape = img.shape

img = resize(img, (SIZE, SIZE, SIZE))
img = (img / np.max(img))

img = nib.Nifti1Image(img, affine)
nib.save(img, "resize.nii")
