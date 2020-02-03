import RNA
writeTo = open('/Users/harrisonlabollita/Arizona State University/Sulc group/data_set/RNA_tgt.txt', 'w+')
readFrom = open('/Users/harrisonlabollita/Arizona State University/Sulc group/data_set/src.txt', 'r')

for line in readFrom:
    seq = line.rstrip()
    writeTo.write(RNA.fold(seq)[0] + '\n')
