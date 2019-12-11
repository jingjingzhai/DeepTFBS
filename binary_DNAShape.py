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
inputFASTA = "/home/malab14/research/00DeepTFBS/dataset/cell_TF/tmp_test/negSet.fa"
record = list(SeqIO.parse(inputFASTA, "fasta"))
resDic = "/home/malab14/research/00DeepTFBS/dataset/cell_TF/tmp_test/featureMat/"
#HelT = np.loadtxt(resDic + "posSet_HelT.txt", dtype="string")
MGW = np.loadtxt(resDic + "negSet_MGW.txt", dtype="string")
ProT = np.loadtxt(resDic + "negSet_ProT.txt", dtype="string")
#Roll = np.loadtxt(resDic + "posSet_Roll.txt", dtype="string")
#EP = np.loadtxt(resDic + "posSet_EP.txt", dtype="string")


trainMat = np.empty(shape = [1, 201, 6])
tt = 0
os.mkdir("./tmp")
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
    curMGW = MGW[tt,:]
    curMGW = curMGW.reshape((1,201,1))
    curProT = ProT[tt,:]
    curProT = curProT.reshape((1,201,1))
    sequence = np.concatenate((sequence, curMGW, curProT), axis=2)
    #print(trainMat.shape)
    #print(sequence.shape)
    trainMat = np.concatenate((trainMat, sequence), axis=0)
    tt = tt + 1
    if tt%1000 == 0 or tt == len(record):
        trainMat = np.delete(trainMat, 0, axis=0)
        trainMat = trainMat.astype(np.float)
        fileName = "./tmp/trainMat_" + str(tt)
        np.save(fileName, trainMat)
        trainMat = np.empty(shape=[1, 201, 6])
        print(tt)



tt = np.arange(1000, len(record), 1000)
tt = np.append(tt, len(record))
resMat = np.empty(shape = [1, 201, 6])
for i in tt:
    curName = "./tmp/trainMat_" + str(i) + ".npy"
    trainMat = np.load(curName)
    resMat = np.concatenate((resMat, trainMat), axis=0)
    print(resMat.shape)

resMat = np.delete(resMat, 0, axis=0)
np.save(resDic+"negSet_binary_DNAshape", resMat)
shutil.rmtree("./tmp")


#training RC-CNN
#install keras from https://github.com/kundajelab/keras/tree/keras_1
from __future__ import print_function
import keras
import numpy as np
from keras.optimizers import SGD
import math
import matplotlib.pyplot as plt
import sys

posMat = np.load("/home/malab14/research/00DeepTFBS/dataset/cell_TF/tmp_test/featureMat/posSet_binary_DNAshape.npy")
negMat = np.load("/home/malab14/research/00DeepTFBS/dataset/cell_TF/tmp_test/featureMat/negSet_binary_DNAshape.npy")

trainIdx = np.load("/home/malab14/research/00DeepTFBS/dataset/cell_TF/tmp_test/featureMat/trainIdx.npy")
valIdx = np.load("/home/malab14/research/00DeepTFBS/dataset/cell_TF/tmp_test/featureMat/valIdx.npy")
testIdx = np.load("/home/malab14/research/00DeepTFBS/dataset/cell_TF/tmp_test/featureMat/testIdx.npy")

posTrain = posMat[trainIdx]
negTrain = negMat[trainIdx]

trainMatrix = np.concatenate((posTrain, negTrain), axis=0)
trainLabel  = [1]*posTrain.shape[0] + [0]*negTrain.shape[0]
trainMatrix = np.delete(trainMatrix, [4,5], axis = 2)


posVal = posMat[valIdx]
negVal = negMat[valIdx]

valMatrix = np.concatenate((posVal, negVal), axis=0)
valMatrix = np.delete(valMatrix, [4,5], axis = 2)
valLabel = [1]*posVal.shape[0] + [0]*negVal.shape[0]



#build a sample model
model = keras.models.Sequential()
model.add(keras.layers.convolutional.RevCompConv1D(input_shape=(201,6),
                                                   nb_filter=32,
                                                   filter_length=11))
model.add(keras.layers.normalization.RevCompConv1DBatchNorm())
model.add(keras.layers.core.Activation("relu"))

#Layer
model.add(keras.layers.convolutional.RevCompConv1D(nb_filter=32,
                                                   filter_length=11))
model.add(keras.layers.normalization.RevCompConv1DBatchNorm())
model.add(keras.layers.core.Activation("relu"))

#Layer
model.add(keras.layers.convolutional.RevCompConv1D(nb_filter=32,
                                                   filter_length=11))
model.add(keras.layers.normalization.RevCompConv1DBatchNorm())
model.add(keras.layers.core.Activation("relu"))
model.add(keras.layers.pooling.MaxPooling1D(pool_length=32))


#Weighted sum
model.add(keras.layers.convolutional.WeightedSum1D(symmetric=False,
                                                   input_is_revcomp_conv=True,
                                                   bias=False,
                                                   init="he_normal"))
model.add(keras.layers.core.DenseAfterRevcompWeightedSum(output_dim=64,
                                                         W_regularizer=keras.regularizers.WeightRegularizer(l2=0)))
model.add(keras.layers.core.Activation("relu"))

#model.add(keras.layers.core.Dropout(0.5))

model.add(keras.layers.core.Dense(output_dim=64))
model.add(keras.layers.core.Activation("relu"))


model.add(keras.layers.core.Dense(output_dim=1))
model.add(keras.layers.core.Activation("sigmoid"))

#lrVec = np.arange(0.00004, 0.00011, 0.00001)
lr = 0.00004
#for lr in lrVec:
sgd = SGD(lr=lr,momentum=0.1)
model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=['accuracy'])
history_callback = model.fit(x=trainMatrix, y=trainLabel, validation_data=(valMatrix, valLabel),
                             batch_size=128, nb_epoch=200)
model.save(filepath=sys.argv[3]+TFID+"_model.h5")

val_acc = np.array(history_callback.history["val_acc"])
train_acc = np.array(history_callback.history["acc"])
res = np.vstack((train_acc, val_acc)).T
fname = saveDir + TFID + ".txt"
header = ("train_acc", "val_acc")
np.savetxt(fname=fname,X=res,delimiter="\t",header="train_acc,val_acc")


f = plt.figure()
plt.plot(range(0,len(train_acc)), train_acc, 'b',range(0, len(val_acc)), val_acc, 'r')
plt.show()
f.savefig(saveDir + TFID + "_evaluation.pdf", bbox_inches='tight')

# PREDICTED_CLASSES = model.predict_classes(testMatrix, batch_size=128, verbose=1)
# PREDICTED_CLASSES = np.reshape(PREDICTED_CLASSES, (len(testLabel)))
# temp = sum(testLabel == PREDICTED_CLASSES)
# temp/len(testLabel)
#
posTestMat = testMatrix[range(0,(testMatrix.shape[0]/2))]
posLabel = testLabel[range(0,(testMatrix.shape[0]/2))]
negTestMat = testMatrix[range(testMatrix.shape[0]/2,testMatrix.shape[0])]
negLabel = testLabel[range(testMatrix.shape[0]/2,testMatrix.shape[0])]

posScore = model.predict(posTestMat, batch_size=128)
#np.save("/home/malab14/research/00DeepTFBS/00results/optimization/cnn_positive_score", posScore)
negScore = model.predict(negTestMat, batch_size=128)
#np.save("/home/malab14/research/00DeepTFBS/00results/optimization/cnn_negative_score", negScore)


negScore = np.load("/home/malab14/research/00DeepTFBS/00results/optimization/cnn_negative_score.npy")
posScore = np.load("/home/malab14/research/00DeepTFBS/00results/optimization/cnn_positive_score.npy")
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








