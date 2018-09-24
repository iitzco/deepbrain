SIZE = 256
LABELS = 139
FILTERED_LABELS = 6

LABEL_MAP = {
        "CSF": 0,
        "WM": 1,
        "GM": 2,
        "BS": 3,
        "CWM": 4,
        "CGM": 5
        }

OUTPUT = "."

ADNI_DATASET_DIR = "/home/ivanitz/Data/adni"

ADNI_TRAIN = "/home/ivanitz/deepbet/data/adni_seg_train.tfrecord"
ADNI_VAL = "/home/ivanitz/deepbet/data/adni_seg_val.tfrecord"

FREQ = {0: 875798010, 1: 1798599147, 2: 2260386338, 3: 76093522, 4: 132872211, 5: 431739109}

FREQ_LIST = [875798010, 1798599147, 2260386338, 76093522, 132872211, 431739109]

FREQ_PROP = [16, 32, 41,  1,  2 , 8]
