SIZE = 128
LABELS = 139
FILTERED_LABELS = 7

LABEL_MAP = {
        "CSF": 1,
        "WM": 2,
        "GM": 3,
        "BS": 4,
        "CWM": 5,
        "CGM": 6
        }

OUTPUT = "."

ADNI_DATASET_DIR = "/home/ivanitz/Data/adni"

ADNI_TRAIN = "/home/ivanitz/deepbet/data/adni_seg_train.tfrecord"
ADNI_VAL = "/home/ivanitz/deepbet/data/adni_seg_val.tfrecord"
