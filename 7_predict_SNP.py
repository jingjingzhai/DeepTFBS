#install keras from https://github.com/kundajelab/keras/tree/keras_1
from __future__ import print_function
import keras
import numpy as np
import sys

'''
Usage:
python 7_predict_SNP.py replaceSNP.txt model.h5 replaceSNP_featureMat.npy positive_featureMat.npy output.txt
'''

SNPPeak = np.loadtxt(sys.argv[1], dtype="str")
model = keras.models.load_model(sys.argv[2])
replaceSNPMat = np.load(sys.argv[3])
posMat = np.load(sys.argv[4])
resMat = open(sys.argv[5], "w")

# SNPPeak = np.loadtxt("/home/malab14/research/00DeepTFBS/00results/06ReplaceSNP/ZFHD_tnt-ATHB24_colamp_a_replaceSNP.txt",
#                      dtype="str")
#
# model = keras.models.load_model("/home/malab14/research/00DeepTFBS/00results/02CNN/model/ZFHD_tnt-ATHB24_colamp_a_model.h5")
# replaceSNPMat = np.load("/home/malab14/research/00DeepTFBS/00results/06ReplaceSNP/featureMat/ZFHD_tnt-ATHB24_colamp_a.npy")
#
# posMat = np.load("/home/malab14/research/00DeepTFBS/00results/01featureMat/positive/ZFHD_tnt-ATHB24_colamp_a.positive.npy")

posScore = model.predict(posMat)
snpScore = model.predict(replaceSNPMat)

# resMat = open("/home/malab14/research/00DeepTFBS/00results/07SNPscore/snp_score.txt", "w")
i = 0
for snp in SNPPeak:
    peakID = int(snp[2].split("_")[1])-1
    snpID = "Chr" + snp[0] + "_" + snp[1]
    score = posScore[peakID] - snpScore[i]
    score = str(score.astype("float")[0])
    output = snp[0] + "\t" + snp[1] + "\t" + str(snpScore[i][0]) + "\t" + str(posScore[peakID][0]) + "\t" + score + "\n"
    resMat.write(output)
    i = i + 1

resMat.close()