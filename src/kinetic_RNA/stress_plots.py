import sys
import numpy as np
import matplotlib.pyplot as plt


dataFile1 = sys.argv[1]
dataFile2 = sys.argv[2]
def getData(filename):
    path = '/Users/harrisonlabollita/Arizona State University/Sulc group/src/kinetic_RNA/'
    data = open(path + filename, 'r')
    lengths = []
    times = []
    times2 = []
    fraction = []
    for line in data:
        line = line.rstrip()
        length, t, t2 = line.split(' ', 2)
        lengths.append(float(length))
        times.append(float(t))
        times2.append(float(t2))
        fraction.append(float(t)/float(t2))
    return lengths, times, times2, fraction



lengths1, times1, times12, fraction1  = getData(dataFile1)
lengths2, times2, times22, fraction2 = getData(dataFile2)

print(np.mean(fraction1))
print(np.mean(fraction2))

#plt.figure()
#plt.scatter(lengths1, times1, color = 'red', label = 'BP in stem = 2')
#plt.scatter(lengths2, times2, color = 'blue', label= 'BP in stem = 3')
#plt.title('Stress Test RNA folder')
#plt.xlabel('Length of Sequence (ntds)')
#plt.ylabel('Runtime (s)')
#plt.grid(True, linewidth = 1, linestyle = ':')
#plt.legend(loc = 'best')
#plt.savefig('stressPlots.eps', format = 'eps')
#fig, ax = plt.subplots()
#ax.grid(True, linewidth=1,linestyle =':')
#ax.scatter(lengths1, times1, color = 'red', label = 'BP in stem = 2')
#ax.set_xlabel('Length of Sequence (ntds)')
#ax.set_ylabel('Runtime (s)')
#ax2 =ax.twinx()
#ax2.set_ylabel('Runtime (s)')
#ax2.scatter(lengths2, times2, color = 'blue', label = 'BP in stem = 3')
#fig.legend(loc = 'upper left')
#plt.savefig('stressTestPlots.eps', format ='eps')
