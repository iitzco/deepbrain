import tensorflow as tf
import numpy as np
from skimage.transform import resize
import os

PB_FILE = os.path.join(os.path.dirname(__file__), "models", "segmenter", "graph.pb")
# CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "models", "segmenter")
CHECKPOINT_DIR = "/Users/ivanitz/Projects/cerebro/deepbrain/deepbrain/models/segmenter/"


class Segmenter:

    def __init__(self):
        self.SIZE = 128
        self.load_ckpt()

    def load_pb(self):
        graph = tf.Graph()
        self.sess = tf.Session(graph=graph)
        with tf.gfile.FastGFile(PB_FILE, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            with self.sess.graph.as_default():
                tf.import_graph_def(graph_def)

        self.img = graph.get_tensor_by_name("import/img:0")
        self.training = graph.get_tensor_by_name("import/training:0")
        self.dim = graph.get_tensor_by_name("import/dim:0")
        self.prob = graph.get_tensor_by_name("import/softmax:0")

    def load_ckpt(self):
        self.sess = tf.Session()
        ckpt_path = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_path))
        saver.restore(self.sess, ckpt_path)

        g = tf.get_default_graph()

        self.img = g.get_tensor_by_name("img:0")
        self.training = g.get_tensor_by_name("training:0")
        self.dim = g.get_tensor_by_name("dim:0")
        self.prob = g.get_tensor_by_name("softmax:0")

    def run(self, image):
        shape = image.shape
        img = resize(image, (self.SIZE, self.SIZE, self.SIZE), mode='constant', anti_aliasing=True)
        img = (img / np.max(img))
        img = np.reshape(img, [1, self.SIZE, self.SIZE, self.SIZE, 1])

        prob = self.sess.run(self.prob, feed_dict={self.training: False, self.img: img}).squeeze()
        prob = resize(prob, (shape), mode='constant', anti_aliasing=True)
        return prob


if __name__ == "__main__":
    import sys
    import nibabel as nib
    import time

    seg = Segmenter()

    img = nib.load(sys.argv[1])

    affine = img.affine
    img = img.get_fdata()

    now = time.time()
    prob = seg.run(img)
    print("Extraction time: {0:.2f} secs.".format(time.time() - now))
    labels = np.argmax(prob, axis=-1)
    out = nib.Nifti1Image(labels, affine)
    nib.save(out, os.path.join(sys.argv[2], "seg.nii"))
