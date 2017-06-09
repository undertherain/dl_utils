import json
import os
import numpy as np


def save_data_json(data, name_file):
    # path = "/"
    # todo bath as basedir
    # if not os.path.isdir(path):
        # os.makedirs(path)
    s = json.dumps(data, ensure_ascii=False, indent=4, sort_keys=True)
    f = open(name_file, 'w')
    print(s, file=f)
    f.close()


def class_to_onehot(vector, cnt_classes):
    res = np.zeros((vector.shape[0], cnt_classes), dtype=np.bool)
    res[np.arange(len(vector)), vector] = 1
    return res
