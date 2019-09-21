import matplotlib.pyplot as plt
import numpy as np
import sys, glob


def rolling(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


#fn = glob.glob("training_attention_50*")
fn = [
    'training_attention_50.log_0.001_0.1',
    'training_attention_100.log_0.0_0.0',
    'training_attention_100.log']

print fn

leg = []
for ff in fn:
    with open(ff) as fid:
        f = [line.strip("\n").split(" ") for line in fid.readlines()]
    try:
        reg = ff.split("_")[3]
        dr = ff.split("_")[4]
    except:
        reg = 0.001
        dr = 0.1
    legend_string = '{} {}'.format(reg,dr)
    leg.append(legend_string)

    keys = f[0]
    f = np.array(f[1:]).T

    print keys, reg, dr

    N = 10
    for k in [1]:
        y = np.asarray(f[k], dtype=float)
        y = rolling(y, N)
        plt.plot(range(len(y)), y)

plt.legend(leg, loc='best', ncol = 2)
plt.show()
