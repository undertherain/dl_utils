import numpy as np
from matplotlib import pyplot as plt


def cmp_experiments(l):
    keys = sorted(l[0].history.keys())
    for i, key in enumerate(keys):
        # ax =
        plt.subplot(2, 2, i+1)
        ymax = []
        for exp in l:
            hist = exp.history
            data = np.array(hist[key])
            data = data[~np.isnan(data)]
            if data.shape[0] > 0:
                ymax.append(data.max())
            plt.plot(np.arange(len(data)) + 1, data, label=exp.descr, linewidth=2)
#        if "loss" in key:
#            print(max(ymax))
#            plt.ylim(0,max(ymax)*1.1)
#        if "acc" in key:
#            plt.ylim(0,1)
        plt.title(key, size=20)
#        if data.max()<1:
#            plt.ylim(0,1)
#        plt.xlim
#        ax.set_yscale("log", nonposy='clip')
        if i == 0:
            legend = plt.legend(loc=1, prop={'size': 18}, bbox_to_anchor=(2.5, 1.8, 0, 0), ncol=2)
            frame = legend.get_frame()
            frame.set_alpha(0)
#            frame.set_facecolor('white')
        plt.tick_params(axis='both', which='major', labelsize=18)


def cmp_loss(l):
    plt.subplot(1, 1, 1)
    ymin = []
    ymax = []
    for exp in l:
        hist = exp.history
        loss = np.array(hist["loss"])
        val_loss = np.array(hist["val_loss"])
        ymin.append(loss.min())
        ymin.append(val_loss.min())
        ymax.append(loss.max())
        ymax.append(val_loss.max())
        # print(ymax)
        plt.scatter(loss, val_loss, label=exp.descr, linewidth=2, alpha=0.7)
        plt.xlabel("loss", size=16)
        plt.ylabel("validation loss", size=16)
    ymax = min(max(ymax), 1)
    plt.ylim(min(ymin), ymax)
    plt.xlim(min(ymin), ymax)
#   legend = plt.legend(loc=1, prop={'size': 18})
    plt.tick_params(axis='both', which='major', labelsize=18)
#   legend=plt.legend(loc=1, prop={'size':18},bbox_to_anchor=(2.1, 1.6, 0, 0))
#   frame = legend.get_frame()
#   frame.set_alpha(0)
#   frame.set_facecolor('white')
