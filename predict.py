from __future__ import print_function
import keras
import numpy as np
from Bio import SeqIO
import os
import sqlite3
import io

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

featureDir = "/home/malab14/research/00DeepTFBS/dataset/whole_genome_featureMat/chr1/featureMat/"
seqDir = "/home/malab14/research/00DeepTFBS/dataset/whole_genome_featureMat/chr1/sequence/"
modelDir = "/home/malab14/research/00DeepTFBS/00results/02CNN/model/"
modelName = os.listdir(modelDir)
fileName = np.loadtxt(fname='/home/malab14/research/00DeepTFBS/dataset/whole_genome_featureMat/chr1/chr1.txt',
                      delimiter="\t", dtype="string")

for TF in modelName:
    model = keras.models.load_model(modelDir + TF)
    tmp = TF.replace('_model.h5', '')
    #resMat = file('chr1_'+ tmp + '.txt', 'a')
    con = sqlite3.connect("/home/malab14/research/test.db", detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("create table test (arr array)")
    for curName in fileName:
        featureMat = np.load(featureDir + curName + ".npy")
        curSeq = list(SeqIO.parse(seqDir + curName, "fasta"))
        seqID = np.array([tt.id for tt in curSeq])
        seqID = seqID.reshape((len(seqID), 1))
        tmpscore = model.predict(featureMat)
        tmpMat = np.concatenate((seqID, tmpscore), axis=1)
        cur.execute("insert into test (arr) values (?)", (tmpMat,))
        #np.savetxt(resMat, tmpMat, delimiter="\t", fmt='%s')

    con.commit()
    print(TF + "is done !")
