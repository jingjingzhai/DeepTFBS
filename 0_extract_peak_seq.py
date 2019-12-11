#! /usr/bin/env/ python
# _*_ coding:utf-8 _*_


import numpy as np
from Bio import SeqIO
import sys
import random

'''
Usage:
python extract_peak_seq.py genomeSequence.fa inputPeak positiveOutput negativeOutput
'''

genomeSeq = SeqIO.parse(sys.argv[1], "fasta")
#genomeSeq = SeqIO.parse("/home/malab14/research/Genomes/Arabidopsis/TAIR10_chr_all.fas", "fasta")
genomeSeqDict = {}
for fasta in genomeSeq:
    name, sequence = fasta.id, fasta.seq.tostring()
    genomeSeqDict[name] = sequence

fileName = sys.argv[2]
#fileName = "/home/malab14/research/00DeepTFBS/dataset/cell_TF/peaks/MYB_tnt/MYB27_colamp_a/chr1-5/chr1-5_GEM_events.narrowPeak"
peakMat = np.loadtxt(fname = fileName, dtype = "string")

#Extracting positive sequence and generating corresponding negative sequence using random shuffling
idx = 1
posSeq = open(sys.argv[3], "w")
#negSeq = open(sys.argv[4], "w")
for peak in peakMat:
    curChr = peak[0][-1]
    curGenomeSeq = genomeSeqDict[curChr]
    seqLen = len(curGenomeSeq)
    startPos = int(peak[1])
    endPos = int(peak[2])
    if startPos == 0:
        peakLen = endPos - startPos
        NString = 'N'*(201-peakLen)
        curSeq = curGenomeSeq[startPos:endPos]
        curSeq = NString + curSeq
        print fileName
    elif endPos > seqLen:
        endPos = seqLen
        peakLen = endPos - startPos
        NString = 'N'*(201-peakLen)
        curSeq = curGenomeSeq[startPos:endPos]
        curSeq = curSeq + NString
        print fileName
    else:
        curSeq = curGenomeSeq[startPos:endPos]

    ranIdx = random.sample(range(0, len(curSeq)), len(curSeq))
    curNegSeq = np.array(list(curSeq))[ranIdx]
    curNegSeq = ''.join(curNegSeq)
    seqID = "peak_" + str(idx)
    idx = idx + 1
    #print(peak[1]+1)
    posSeq.write(">" + seqID + "_" + str(curChr) + ":" +
                 str(int(peak[1])+1) + "-" + str(peak[2]) +
                 "\n" + curSeq + "\n")
    #negSeq.write(">" + seqID + "_" + str(curChr) + ":" +
                 #str(int(peak[1])+1) + "-" + str(peak[2]) + "\n"
                 #+ curNegSeq + "\n")

posSeq.close()
#negSeq.close()