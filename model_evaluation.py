import numpy as np
import os

fileName = os.listdir("/home/malab14/research/00DeepTFBS/00results/05gcForest/evaluate/")
measures = ['Sn', 'Sp', 'Pr', 'Acc', 'Fscore', 'MCC']

resMat = open("/home/malab14/research/00DeepTFBS/00results/05gcForest/gcForest_evaluation.txt", "w")
header = "\t".join(measures)
resMat.write(header + "\n")
#resMat = np.empty((1,6), dtype="str")
for file in fileName:
    curRes = np.load("/home/malab14/research/00DeepTFBS/00results/05gcForest/evaluate/" + file)
    curTF = file.replace("_evaluate.npy", "")
    curList = []
    curList.append(curTF)
    for j in measures:
        curList.append(str(curRes.item().get(j)))
    curList = "\t".join(curList)
    resMat.write(curList + "\n")

resMat.close()
