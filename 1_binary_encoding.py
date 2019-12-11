from Bio import SeqIO
import sys
import os
import shutil
import numpy as np
import re

'''
Usage:
python binary_encoding.py input.fa output
'''

#Convert to sequence to binary
inputFASTA = sys.argv[1]
record = list(SeqIO.parse(inputFASTA, "fasta"))

trainMat = np.empty(shape = [1, 201, 4])
tt = 0
for fasta in record:
    name, sequence = fasta.id, fasta.seq.tostring()
    sequence = re.sub(r'[^ACGT]', "0 0 0 0 ", sequence)
    sequence = sequence.replace('A', "1 0 0 0 ")
    sequence = sequence.replace('C', "0 1 0 0 ")
    sequence = sequence.replace('G', "0 0 1 0 ")
    sequence = sequence.replace('T', "0 0 0 1 ")
    sequence = sequence.split(" ")
    del(sequence[len(sequence)-1])
    sequence = np.array(sequence)
    shape = (1, len(sequence)/4, 4)
    sequence = sequence.reshape(shape)
    #print(trainMat.shape)
    #print(sequence.shape)
    trainMat = np.concatenate((trainMat, sequence), axis=0)

trainMat = np.delete(trainMat, 0, axis=0)
trainMat = trainMat.astype(np.float)
np.save(sys.argv[2], trainMat)
