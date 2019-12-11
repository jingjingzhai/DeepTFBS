import h5py
import numpy as np

h5Name = "/home/malab14/research/103.hdf5"
gwas = h5py.File(h5Name, 'r')

chromosome = ['chr1','chr2','chr3','chr4','chr5']
resDic = "/home/malab14/research/tmp/"
for chr in chromosome:
    curPos = gwas["/pvalues/"+chr+"/positions"][:]
    curScore = gwas["/pvalues/"+chr+"/scores"][:]
    curPos = curPos.astype(int)
    #curPos = curPos.reshape((len(curPos), 1))
    #curScore = curScore.reshape((len(curScore),1))
    np.savetxt(resDic + chr + "_position.txt", curPos, fmt = "%d")
    np.savetxt(resDic + chr + "_score.txt", curScore)

