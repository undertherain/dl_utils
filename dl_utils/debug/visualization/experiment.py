import os
import json
import numpy as np

dir_root = "/work/alex/data/DL_outs/Kaggle/2017.05_Planet/"


def json_to_optimizer_str(s):
    j = json.loads(s)
    res = ""
    if "optimizer" in j:
        res += j["optimizer"]
    if "learning_rate" in j:
        res += "."+str(j["learning_rate"])
    if "batch_size" in j:
        res += ":bs"+str(j["batch_size"])
    if "hostname" in j:
        res += "("+str(j["hostname"])+")"
    return res


class experiment:
    def __init__(self, path):
        self.path = os.path.join(dir_root, path)
        if "val_loss" in self.history:
            self.best = min(self.history["val_loss"])
            self.id_epoch_best = np.argmin(self.history["val_loss"])+1
        else:
            self.best = min(self.history["loss"])
            self.id_epoch_best = np.argmin(self.history["loss"])+1
        self.load_options()
        self.descr = self.name + "|" + self.str_arch + "|" + self.str_opt
        if "shape_x_train" in self.params:
            self.descr = str(self.params["shape_x_train"][0]) + "x"+self.descr

    def load_options(self):
        if os.path.isfile(os.path.join(self.path, "options.json")):
            with open(os.path.join(self.path, "options.json")) as f:
                s = f.read()
                self.params = json.loads(s)
                self.str_opt = json_to_optimizer_str(s)
        else:
            self.str_opt = ""
            self.params = {}

    def best_loss(self):
        return self.best


class experiment_chainer(experiment):
    def __init__(self, path):
        with open(os.path.join(path, "log")) as f:
            str_log = f.read()
        log = json.loads(str_log)
        self.history = {}
        self.history["loss"] = np.array([i["main/loss"] for i in log])
        # self.history["loss"] = np.clip(self.history["loss"],a_min=0,a_max=5)
        if "main/accuracy" in log[0]:
            self.history["acc"] = np.array([i["main/accuracy"] for i in log])
            self.history["val_acc"] = np.array([i["validation/main/accuracy"] for i in log])
        self.history["val_loss"] = np.array([i["validation/main/loss"] for i in log])
        # self.history["val_loss"] = np.clip(self.history["val_loss"],a_min=0,a_max=5)
        self.name = path.split("/")[-1]
        self.str_arch = "chnr"
        self.str_opt = "dummy opt"
        super(experiment_chainer, self).__init__(path)


def load_experiments(path_root):
    experiments = []
    for name_exp in os.listdir(path_root):
        path = os.path.join(path_root, name_exp)
        if name_exp[0] == "!":
            continue
        try:
            if os.path.isfile(os.path.join(path, "log")):
                exp = experiment_chainer(path)
            else:
                # exp = experiment_keras(name_exp)
                pass
            experiments.append(exp)
        except BaseException as e:
            print(name_exp, "seems to be corrupt ")
            print(e)
    # experiments.sort(key=lambda x:x.best_loss())
    experiments.sort(key=lambda x: x.name)
    return experiments


def main():
    print("test")
    experiments = load_experiments(dir_root)
    print(len(experiments))


if __name__ == '__main__':
    main()
