#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import math
import pickle
import sys

'''
Usage:
python 4_train_RF.py featureMat_directory TFID saveDir
'''

#Load training data
resDic = sys.argv[1]
TFID = sys.argv[2]
saveDir = sys.argv[3]

trainMatrix = np.load(resDic + TFID + "_trainMatrix.npy")
trainLabel = np.load(resDic + TFID + "_trainLabel.npy")

valMatrix = np.load(resDic + TFID + "_valMatrix.npy")
valLabel = np.load(resDic + TFID + "_valLabel.npy")


trainMat = np.concatenate((trainMatrix, valMatrix), axis=0)
trainMat = np.reshape(trainMat, (trainMat.shape[0],trainMat.shape[1]*trainMat.shape[2]))
trainY = list(trainLabel) + list(valLabel)

trainY = np.array(trainY)


testMatrix = np.load(resDic + TFID + "_testMatrix.npy")
testMatrix = np.reshape(testMatrix, (testMatrix.shape[0],testMatrix.shape[1]*testMatrix.shape[2]))
testLabel = np.load(resDic + TFID + "_testLabel.npy")


posTestMat = testMatrix[range(0,(testMatrix.shape[0]/2))]
posLabel = testLabel[range(0,(testMatrix.shape[0]/2))]
negTestMat = testMatrix[range(testMatrix.shape[0]/2,testMatrix.shape[0])]
negLabel = testLabel[range(testMatrix.shape[0]/2,testMatrix.shape[0])]

RFclf = RandomForestClassifier(n_estimators = 500, criterion = 'gini', n_jobs=10)
RFclf.fit(trainMat, trainY)
with open(saveDir+TFID+"_RFmodel.h5", 'wb') as f:
    pickle.dump(RFclf, f)

posScore = RFclf.predict(posTestMat)
negScore = RFclf.predict(negTestMat)
def evalModel(posScore, negScore, threshold = 0.5, beta = 2):
    TP = float(sum(posScore > threshold))
    TN = float(sum(negScore <= threshold))
    FP = float(len(posScore)-TP)
    FN = float(len(negScore)-TN)
    res = {}
    res['Sn'] = TP/(TP + FN)
    res['Sp'] = TN/(TN + FP)
    res['Pr'] = TP/(TP + FP)
    res['Acc'] = (TP+TN)/(TP+TN+FP+FN)
    res['Fscore'] = ((1+beta*beta)*res['Pr']*res['Sn'])/(beta*beta*res['Pr']+res['Sn'])
    res['MCC']=(TP*TN-FP*FN)/math.sqrt(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    return res

res = evalModel(posScore=posScore,negScore=negScore)
np.save(saveDir+TFID+"_evaluate", res)


# SVCclf = SVC()
# SVCclf.fit(trainMat, trainY)
# with open('/home/malab14/research/00DeepTFBS/00results/optimization/SVCModel', 'wb') as f:
#     pickle.dump(SVCclf, f)
# posScore = SVCclf.predict(posTestMat)
# negScore = SVCclf.predict(negTestMat)
# evalModel(posScore=posScore,negScore=negScore)
