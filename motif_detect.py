import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from keras import backend as K
import os, sys

modelName = sys.argv[1]
trainMat = sys.argv[2]
trainLabel = sys.argv[3]
outDir = sys.argv[4]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model = keras.models.load_model(modelName)
#reverse complement layer
firstLayer = model.layers[0].get_weights()
weights = np.reshape(firstLayer[0],(32,11,4))


inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions


featureMat = np.load(trainMat)
label = np.load(trainLabel)

def extractSeq(index, inputMat):
    res = {}
    for i in index:
        curMat = inputMat[0][range(i,i+11)]
        curSeq = np.array(["".join(list(j.astype(int).astype(str))) for j in curMat])
        curSeq[curSeq == '0001'] = 'T'
        curSeq[curSeq == '0010'] = 'G'
        curSeq[curSeq == '0100'] = 'C'
        curSeq[curSeq == '1000'] = 'A'
        curSeq[curSeq == '0000'] = 'N'
        curSeq = "".join(list(curSeq))
        curID = str(i) +"-" + str(i+11)
        res[curID] = curSeq
    return res

def writeFasta(seqDic, out):
    res = open(out, "a+")
    for i in seqDic.keys():
        res.write(">" + i + "\n" + seqDic[i] + "\n")
    res.close()


for j in range(0,64):
    curName = outDir + "kernel" + str(j+1) + ".fa"
    posVec = np.where(label == 1)[0]
    print(str(j) + ":" + str(len(posVec)))
    for i in posVec:
        curInput = featureMat[i]
        curInput = curInput[np.newaxis, ...]
        layer_outs = [func([curInput, 1.]) for func in functors]
        res = layer_outs[2][0][0]
        curOut = res[:,j]
        curOut = pd.Series(list(curOut), index=range(0, 191))
        curOut = pd.Series.sort_values(curOut)[::-1]
        curIdx = curOut[0:5].index
        resSeq = extractSeq(index=curIdx, inputMat=curInput)
        writeFasta(seqDic=resSeq,out=curName)
