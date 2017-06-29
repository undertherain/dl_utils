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

# cmp_experiments(experiments[-2:])
