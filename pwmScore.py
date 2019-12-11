import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import math
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def convertSeq(inputMat):
    res = {}
    for c, i in enumerate(range(len(inputMat)), 1):
        curMat = inputMat[i]
        curSeq = np.array(["".join(list(j.astype(int).astype(str))) for j in curMat])
        curSeq[curSeq == '0001'] = 'T'
        curSeq[curSeq == '0010'] = 'G'
        curSeq[curSeq == '0100'] = 'C'
        curSeq[curSeq == '1000'] = 'A'
        curSeq[curSeq == '0000'] = ''
        curSeq = "".join(list(curSeq))
        curID = 'seq' + str(c)
        res[curID] = curSeq
    return res

def bgPWM(pwmMat):
    data = {'A':[0.25] * len(pwmMat),
            'C':[0.25] * len(pwmMat),
            'G': [0.25] * len(pwmMat),
            'T': [0.25] * len(pwmMat)
            }
    bg = pd.DataFrame(data = data)
    return bg

def pwmScore(pwmMat, inputSeq):
    bg = bgPWM(pwmMat)
    seqID = inputSeq.keys()
    k = len(pwmMat) #motif length
    loopID = range((201- k + 1))
    resDic = {}
    for i in seqID:
        curSeq = inputSeq[i]
        resScore = []
        for c, j in enumerate(loopID):
            tmpSeq = curSeq[j:(j+k)]
            curScore = sum([pwmMat[base][count]/bg[base][count] for count, base in enumerate(tmpSeq)])
            if(curScore != 0):
                curScore = math.log10(curScore)
            resScore.append(curScore)
        resScore = max(resScore)
        resDic[i] = resScore
    results = pd.Series(resDic)
    return results


curTF = "NAC_tnt-ANAC053_col"
pwm = pd.read_csv('/home/malab14/research/00DeepTFBS/dataset/cell_TF/pwm/' + curTF + ".txt",
                  sep = "\t", header = None)
pwm = pwm[pwm.columns[0:4]]
pwm.columns = ['A', 'C', 'G', 'T']

seqDic = "/home/malab14/research/00DeepTFBS/00results/01featureMat/split_samples/"
testSeq = np.load(seqDic + curTF + "_testMatrix.npy")
testLabel = np.load(seqDic + curTF + "_testLabel.npy")

posSeq = testSeq[np.where(testLabel == 1)]
posSeq = convertSeq(posSeq)
negSeq = testSeq[np.where(testLabel == 0)]
negSeq = convertSeq(negSeq)

posScorePWM = pwmScore(pwm, posSeq)
negScorePWM = pwmScore(pwm, negSeq)

ScoreWPM = np.concatenate((posScorePWM, negScorePWM), axis = 0)
Label = [1]*len(posScorePWM) + [0]*len(negScorePWM)
fprPWM, tprPWM, thresholdsPWM = roc_curve(Label, ScoreWPM)
aucPWM = roc_auc_score(Label, ScoreWPM)

###Decision tree
DTDic = "/home/malab14/research/00DeepTFBS/00results/04DT/score/"
posScoreDT = np.load(DTDic + curTF + "_posScore.npy")
negScoreDT = np.load(DTDic + curTF + "_negScore.npy")
scoreDT = np.concatenate((posScoreDT, negScoreDT), axis = 0)
frpDT, tprDT, thresholdsDT = roc_curve(Label, scoreDT)
aucDT = roc_auc_score(Label, scoreDT)

###XGBoost
XGBoostDic = "/home/malab14/research/00DeepTFBS/00results/04XGBoost/score/"
posScoreXGB = np.load(XGBoostDic + curTF + "_posScore.npy")
negScoreXGB = np.load(XGBoostDic + curTF + "_negScore.npy")
scoreXGB = np.concatenate((posScoreXGB, negScoreXGB), axis = 0)
fprXGB, tprXGB, thresholdsXGB = roc_curve(Label, scoreXGB)
aucXGB = roc_auc_score(Label, scoreXGB)

###CNN
CNNDic = "/home/malab14/research/00DeepTFBS/00results/02CNN/threshold/"
posScoreCNN = np.load(CNNDic + curTF + "_posScore.npy")
negScoreCNN = np.load(CNNDic + curTF + "_negScore.npy")
scoreCNN = np.concatenate((posScoreCNN, negScoreCNN), axis=0)
fprCNN, tprCNN, thresholdsCNNB = roc_curve(Label, scoreCNN)
aucCNN = roc_auc_score(Label, scoreCNN)

print("Decision tree (AUC): ", aucDT)
print("XGBoost (AUC): ", aucXGB)
print("PWM (AUC): ", aucPWM)
print("CNN (AUC): ", aucCNN)

f = plt.figure()
plt.plot(fprPWM, tprPWM, 'b')
plt.plot(fprDT, tprDT, 'k')
plt.plot(fprXGB, tprXGB, 'r')
plt.plot(fprCNN, tprCNN, 'r')
f.savefig("roc.pdf", bbox_inches = 'tight')



