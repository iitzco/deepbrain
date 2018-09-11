import tensorflow as tf
import os
import sys
import nibabel as nib
import numpy as np
import random
import re
from skimage.transform import resize


from const import *

class Extractor:

    def __init__(self):
        self.load()

    def load(self):
        self.sess = tf.Session()
        ckpt_path = tf.train.latest_checkpoint("./models/")
        saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_path))
        saver.restore(self.sess, ckpt_path)

        g = tf.get_default_graph()

        self.img = g.get_tensor_by_name("img:0")
        self.training = g.get_tensor_by_name("training:0")
        self.dim = g.get_tensor_by_name("dim:0")
        self.prob = g.get_tensor_by_name("prob:0")
        self.pred = g.get_tensor_by_name("pred:0")

    def run(self, image):
        shape = image.shape
        img = resize(image, (SIZE, SIZE, SIZE))
        img = (img / np.max(img))
        img = np.reshape(img, [1, SIZE, SIZE, SIZE, 1])

        prob = self.sess.run(self.prob, feed_dict={self.training: False, self.img: img}).squeeze()
        prob = resize(prob, (shape))
        return prob


def run(model):
    pass


if __name__ == "__main__":
    img = nib.load(sys.argv[1])
    affine = img.affine
    img = img.get_fdata()

    extractor = Extractor()
    
    import time

    now = time.time()
    prob = extractor.run(img)
    print(time.time() - now)
    brain = 1 * (prob > 0.3)

    brain = nib.Nifti1Image(brain, affine)
    nib.save(brain, "brain.nii")
