from Bio import SeqIO
import numpy as np
import sys

'''
Usage:
python 6_replace_snp.py intersectSNP.txt peakSequence.fa peakMatrix.txt output.fa
'''

intersectSNPDir = sys.argv[1]
peakSeqDir = sys.argv[2]
peakDir = sys.argv[3]
outDir = sys.argv[4]

SNP = np.loadtxt(intersectSNPDir, dtype="str")

#peakName = "/home/malab14/research/00DeepTFBS/00results/00peakSeq/positive/ZFHD_tnt-ATHB34_colamp_a.positive.fa"
peakSeq = SeqIO.parse(peakSeqDir, "fasta")

peakSeqDict = {}
for fasta in peakSeq:
    name, sequence = fasta.id, fasta.seq.tostring()
    peakSeqDict[name] = sequence


peakMat = np.loadtxt(peakDir, dtype="str")
startPos = peakMat[:, 1].astype(int)
endPos = peakMat[:, 2].astype(int)
chrVec = peakMat[:, 0]
resSeq = open(outDir, "w")
resMat = open(sys.argv[5], "w")
for line in SNP:
    curPos = int(line[2])
    curChr = line[0][-1]
    tt = np.intersect1d(np.where(startPos <= curPos),
                         np.where(endPos >= curPos))
    idx = np.intersect1d(tt, np.where(chrVec == "chr"+str(curChr)))
    idx = idx.astype(int)
    for j in idx:
        peak = peakMat[j]
        curStart = int(peak[1])
        curEnd = int(peak[2])
        seqID = "peak_" + str(j + 1)
        peakID = seqID + "_" + str(curChr) + ":" + str(int(peak[1]) + 1) + "-" + str(peak[2])
        curSeq = peakSeqDict[peakID]
        RP = curPos - curStart - 1
        Ref = line[3]
        Alt = line[4]
        if curStart != 0:
            if Ref == curSeq[RP]:
                if len(Ref) == 1 and len(Alt) == 1:
                    curSeq = list(curSeq)
                    curSeq[RP] = Alt
                    curSeq = ''.join(curSeq)
                    resSeq.write(">" + peakID + ";SNP_position:" + str(
                        curPos) + ";Ref:" + Ref + ";Alt:" + Alt + "\n" + curSeq + "\n")
                    resMat.write(str(curChr) + "\t" + str(curPos) + "\t" + peakID + "\n")
        else:
            if Ref == curSeq[RP]:
                if len(Ref) == 1 and len(Alt) == 1:
                    seqLen = curEnd - curStart
                    RP = RP + (201 - seqLen)
                    print(Ref)
                    curSeq = list(curSeq)
                    print(curSeq[RP])
                    curSeq[RP] = Alt
                    curSeq = ''.join(curSeq)
                    resSeq.write(">" + peakID + ";SNP_position:" + str(
                        curPos) + ";Ref:" + Ref + ";Alt:" + Alt + "\n" + curSeq + "\n")
                    resMat.write(str(curChr) + "\t" + str(curPos) + "\t" + peakID + "\n")
resSeq.close()
resMat.close()





# t = 0
# posSeq = open("/home/malab14/research/00DeepTFBS/00results/00peakSeq/positive_replaceSNP.fa", "w")
# for fasta in peakSeq:
#     name, sequence = fasta.id, fasta.seq.tostring()
#     sequence = list(sequence)
#     curChr = re.split(r'[_:-]', name)[2]
#     curChr = "Chr" + curChr
#     Start = int(re.split(r'[_:-]', name)[3])
#     End = int(re.split(r'[_:-]', name)[4])
#     idx = np.where(SNP[:,3] == curChr)
#     #Find first column contain "Chr"
#     #idx = np.flatnonzero(np.core.defchararray.find(SNP[:,0], Chr)!=-1)
#     curSNP = SNP[idx]
#     position = curSNP[:,4].astype(int)
#     resIdx = np.intersect1d(np.where(position >= Start), np.where(position <= End))
#     t = t+1
#     if resIdx.shape[0] == 0:
#         continue
#     else:
#         resSNP = curSNP[resIdx]
#         for record in resSNP:
#             curPos = int(record[4])
#             sequence[curPos-Start] = record[2]
#
#         sequence = ''.join(sequence)
#         print(t)
#
#         posSeq.write(">" + name + ";" + ';'.join(list(resSNP[:,0].astype(str))) + "\n" + sequence + "\n")
#
#
# posSeq.close()
#
#
