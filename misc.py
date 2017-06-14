import datetime
import os
import sys
import shutil
import json


def get_time_str():
    d = datetime.datetime.now()
    s = d.strftime("%y.%m.%d_%H.%M.%S")
    return s


def save_data_json(data, name_file):
    # path = "/"
    # todo path as basedir
    # if not os.path.isdir(path):
        # os.makedirs(path)
    s = json.dumps(data, ensure_ascii=False, indent=4, sort_keys=True)
    f = open(name_file, 'w')
    print(s, file=f)
    f.close()


def save_options(options, path):
    save_data_json(options, os.path.join(path, "options.json"))


def save_code(path):
    os.makedirs(path, exist_ok=True)
    shutil.copy2(sys.argv[0], os.path.join(path, sys.argv[0]))
    # todo: detect local imports
