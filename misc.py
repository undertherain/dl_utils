import datetime
import os
import sys
import shutil


def get_time_str():
    d = datetime.datetime.now()
    s = d.strftime("%y.%m.%d_%H.%M.%S")
    return s


def save_code(path):
    os.makedirs(path, exist_ok=True)
    shutil.copy2(sys.argv[0], os.path.join(path, sys.argv[0]))
    # todo: detect local imports
