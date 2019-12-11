import matplotlib
matplotlib.use('Agg')
import keras
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import pickle

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

CNN = "/home/malab14/research/00DeepTFBS/00results/02CNN/model/"
RF = "/home/malab14/research/00DeepTFBS/00results/03RF/model/"
SVM = "/home/malab14/research/00DeepTFBS/00results/04SVM/"
resDic = "/home/malab14/research/00DeepTFBS/00results/01featureMat/split_samples/"
figureDic = "/home/malab14/research/00DeepTFBS/01results/ROC/"
TFID = np.loadtxt(fname="/home/malab14/research/00DeepTFBS/dataset/train_TF", dtype="string")


AUCMat = open("/home/malab14/research/00DeepTFBS/01results/AUCMatrix.txt", "w")
AUCMat.write("CNN\tRF\tSVM\n")
for curTF in TFID:
    testMatrix = np.load(resDic + curTF + "_testMatrix.npy")
    testLabel = np.load(resDic + curTF + "_testLabel.npy")
    CNNmodel = keras.models.load_model(CNN + curTF + "_model.h5")
    CNNScore = CNNmodel.predict(testMatrix)
    testMatrix = np.reshape(testMatrix, (testMatrix.shape[0], testMatrix.shape[1] * testMatrix.shape[2]))
    RFModel = pickle.load(open(RF + curTF + "_RFmodel.h5", "rb"))
    RFScore = RFModel.predict_proba(testMatrix)[:, 1]
    SVMModel = pickle.load(open(SVM + curTF + "_SVMmodel.h5", "rb"))
    SVMScore = SVMModel.predict_proba(testMatrix)[:, 1]

    CNNfpr, CNNtpr, CNNthresholds = roc_curve(testLabel, CNNScore, pos_label=1)
    RFfpr, RFtpr, RFthresholds = roc_curve(testLabel, RFScore, pos_label=1)
    SVMfpr, SVMtpr, SVMthresholds = roc_curve(testLabel, SVMScore, pos_label=1)
    RFAUC = roc_auc_score(testLabel, RFScore)
    SVMAUC = roc_auc_score(testLabel, SVMScore)
    CNNAUC = roc_auc_score(testLabel, CNNScore)

    AUCMat.write(str(CNNAUC) + "\t" + str(RFAUC) + "\t" + str(SVMAUC) + "\n")
    f = plt.figure()
    plt.plot(CNNfpr, CNNtpr, "r-", label = "CNN:"+str(CNNAUC))
    plt.plot(RFfpr, RFtpr, "b-", label="RF:" + str(RFAUC))
    plt.plot(SVMfpr, SVMtpr, "g-", label="SVM:" + str(SVMAUC))
    plt.legend()
    #plt.plot(CNNfpr, CNNtpr, "red", RFfpr, RFtpr, "blue", SVMfpr, SVMtpr, "grey")
    f.savefig(figureDic + curTF + "_ROC.pdf", bbox_inches='tight')
AUCMat.close()
#SVMModel = pickle.load(open(SVM + TFID + "_SVMmodel.h5", "rb"))
#SVMScore = SVMModel.predict(testMatrix)










