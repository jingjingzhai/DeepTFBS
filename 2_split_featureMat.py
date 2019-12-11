import numpy as np
import random
import sys

'''
Usage:
python split_featureMat.py posFeatureMat.npy negFeatureMat.npy outDir
'''


posMat = np.load(sys.argv[1])
negMat = np.load(sys.argv[2])
outDir = sys.argv[3]
namePre = sys.argv[1].split("/")[-1]
namePre = namePre.split(".")[0]



#Generating index for training, validation and test
samIdx = range(0, posMat.shape[0])
trainIdx = random.sample(range(0, posMat.shape[0]), int(posMat.shape[0]*0.7))
remainIdx = set(samIdx) - set(trainIdx)
remainIdx = list(remainIdx)

remainIdx = np.array(remainIdx)
valIdx = remainIdx[random.sample(range(0,len(remainIdx)), int(len(remainIdx)*0.5))]
testIdx = set(remainIdx) - set(valIdx)

valIdx = list(valIdx)
testIdx = list(testIdx)
#Generating index for training, validation and test
np.save("/home/malab14/research/00DeepTFBS/dataset/cell_TF/tmp_test/featureMat/trainIdx", trainIdx)
np.save("/home/malab14/research/00DeepTFBS/dataset/cell_TF/tmp_test/featureMat/valIdx", valIdx)
np.save("/home/malab14/research/00DeepTFBS/dataset/cell_TF/tmp_test/featureMat/testIdx", testIdx)

#training matrix and label
trainPosMat = posMat[trainIdx]
trainNegMat = negMat[trainIdx]
trainMat = np.concatenate((trainPosMat, trainNegMat), axis=0)
np.save(outDir + "/" + namePre + "_trainMatrix", trainMat)
trainLabel = [1]*trainPosMat.shape[0] + [0]*trainNegMat.shape[0]
np.save(outDir + "/" + namePre + "_trainLabel", trainLabel)
#training matrix and label

#Validation matrix and label
valPosMat = posMat[valIdx]
valNegMat = negMat[valIdx]
valMat = np.concatenate((valPosMat, valNegMat), axis=0)
np.save(outDir + "/" + namePre + "_valMatrix", valMat)
valLabel = [1]*valPosMat.shape[0] + [0]*valNegMat.shape[0]
np.save(outDir + "/" + namePre + "_valLabel", valLabel)
#Validation matrix and label

#test matrix and label
testPosMat = posMat[testIdx]
testNegMat = negMat[testIdx]
testMat = np.concatenate((testPosMat, testNegMat), axis=0)
np.save(outDir + "/" + namePre + "_testMatrix", testMat)
testLabel = [1]*testPosMat.shape[0] + [0]*testNegMat.shape[0]
np.save(outDir + "/" + namePre + "_testLabel", testLabel)
#test matrix and label

