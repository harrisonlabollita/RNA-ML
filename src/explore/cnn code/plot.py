import numpy as np
import matplotlib.pyplot as plt

def plotmodel(epochs, loss, val_loss, title):
    epo = np.xrange(1, epochs)
    plt.figure()
    plt.title(title)
    plt.grid(True, linestyle = ':', linewidth = 1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epo, loss, linewidth = 1, label = 'Training Loss')
    plt.plot(epo, val_loss, linewidth = 1, label = 'Val Loss')
    plt.legend(loc='best')
    plt.show()
