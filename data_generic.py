import json
import os
import numpy as np


def class_to_onehot(vector, cnt_classes):
    res = np.zeros((vector.shape[0], cnt_classes), dtype=np.bool)
    res[np.arange(len(vector)), vector] = 1
    return res
