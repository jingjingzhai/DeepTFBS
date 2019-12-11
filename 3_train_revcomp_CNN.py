#install keras from https://github.com/kundajelab/keras/tree/keras_1
from __future__ import print_function
import keras
import numpy as np
from keras.optimizers import SGD
import math
import matplotlib.pyplot as plt
import sys

'''
Usage:
python 3_train_revcomp_CNN.py featureMat_directory TFID saveDir
'''

#Load training data
resDic = sys.argv[1]
TFID = sys.argv[2]
saveDir = sys.argv[3]
trainMatrix = np.load(resDic + TFID + "_trainMatrix.npy")
trainLabel = np.load(resDic + TFID + "_trainLabel.npy")

valMatrix = np.load(resDic + TFID + "_valMatrix.npy")
valLabel = np.load(resDic + TFID + "_valLabel.npy")

testMatrix = np.load(resDic + TFID + "_testMatrix.npy")
testLabel = np.load(resDic + TFID + "_testLabel.npy")

#build a sample model
model = keras.models.Sequential()
model.add(keras.layers.convolutional.RevCompConv1D(input_shape=(201,4),
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
sgd = SGD(lr=lr)
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
np.save(saveDir + TFID + "_evaluate", res)

