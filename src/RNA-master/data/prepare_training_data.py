import glob, sys, os
import numpy as np

def read_file(fileName):
    record = {}
    with open(fileName, "r") as fid:
        for line in fid.readlines()[1:]:
            l = [x for x in line.split(' ') if x != '']
            n = int(l[0])
            base = l[1]
            partner = int(l[4])
            record[n] = [base,partner]
    seq = "".join([record[x][0] for x in sorted(record.keys())])
    return seq, record

if __name__ == '__main__':

    seq_files = glob.glob('data/seed_structures/*')

    bonds = []
    lens = []
    seqs = []
    comp_seqs = []
    for dataFile in seq_files:
        seq, record = read_file(dataFile)
        hb_comp_seq = ""
        print_seq = False
        for b in sorted(record.keys()):
            base = record[b][0]
            pair = record[b][1]
            if pair:

                if base == 'A' and record[pair][0] == 'C':
                    print_seq = True
                if base == 'C' and record[pair][0] == 'A':
                    print_seq = True

                hb_comp_seq += str(record[pair][0])
            else:
                hb_comp_seq += str('X')
        bonds.append(len([x for x in record.values() if x[1]]) / 2)
        lens.append(len(seq))
        seqs.append(seq)
        comp_seqs.append(hb_comp_seq)

        if print_seq:
            #print(dataFile)
            #print(seq)
            #print(hb_comp_seq)
            print_seq = False

    lens = np.array(lens)
    print("Total strands", len(lens))
    print("Shortest strand", min(lens))
    print("Longest strand", max(lens))
    print("Number that have size < 1000", len(np.where(lens[:]<1000)[0]))
    print("Length mean", np.mean(lens))
    print("Length std", np.std(lens))
    print("Average bonds", np.mean(bonds))

    if os.path.isfile('data/training/sequences.txt'):
        os.system('rm data/training/sequences.txt')
    with open('data/training/sequences.txt', 'w') as fid:
        for seq,comp_seq in zip(seqs,comp_seqs):
            fid.write("%s %s\n" % (seq,comp_seq))
