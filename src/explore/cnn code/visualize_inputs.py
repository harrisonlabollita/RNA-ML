import numpy as np
import matplotlib.pyplot as plt
import importData as data

source = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/test_src.txt'
target = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/test_tgt.txt'



sequences = data.getRNA(source)
dotBrackets = data.getDotBrackets(target)

rna2D = data.seqs2matrices(sequences)


# Visualize

x_labels = sequences[1]
y_labels = sequences[1]
fig, ax = plt.subplots()
ax.imshow(rna2D[1])

ax.set_xticks(np.arange(rna2D[1].shape[1]) + 0.15, minor=False)
ax.set_yticks(np.arange(rna2D[1].shape[0]) + 0.15, minor=False)
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.xaxis.set_ticks_position('both') # THIS IS THE ONLY CHANGE
ax.tick_params(axis='both', which='both', length=0)
ax.set_xticklabels(x_labels, minor=False)
ax.set_yticklabels(y_labels, minor=False)
plt.show()


for i in range(5):
    fig, ax = plt.subplots()
    x_labels = sequences[i]
    y_labels = sequences[i]
    plot = ax.imshow(rna2D[i])
    ax.set_xticks(np.arange(rna2D[i].shape[1]) + 0.15, minor=False)
    ax.set_yticks(np.arange(rna2D[i].shape[0]) + 0.15, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_ticks_position('both') # THIS IS THE ONLY CHANGE
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels(x_labels, minor=False)
    ax.set_yticklabels(y_labels, minor=False)
    fig.colorbar(plot, ax = ax)
    plt.show()
