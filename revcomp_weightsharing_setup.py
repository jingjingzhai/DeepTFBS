#install keras from https://github.com/kundajelab/keras/tree/keras_1
from __future__ import print_function
import keras
import numpy as np
np.random.seed(1)

#build a sample model
model = keras.models.Sequential()
model.add(keras.layers.convolutional.RevCompConv1D(input_shape=(100,4),
                                                   nb_filter=10,
                                                   filter_length=11))
model.add(keras.layers.normalization.RevCompConv1DBatchNorm())
model.add(keras.layers.core.Activation("relu"))
model.add(keras.layers.convolutional.RevCompConv1D(nb_filter=10,
                                                   filter_length=11))
model.add(keras.layers.normalization.RevCompConv1DBatchNorm())
model.add(keras.layers.core.Activation("relu"))
model.add(keras.layers.convolutional.RevCompConv1D(nb_filter=10,
                                                   filter_length=11))
model.add(keras.layers.normalization.RevCompConv1DBatchNorm())
model.add(keras.layers.core.Activation("relu"))
model.add(keras.layers.pooling.MaxPooling1D(pool_length=10))


model.add(keras.layers.convolutional.WeightedSum1D(symmetric=False,
                                                   input_is_revcomp_conv=True,
                                                   bias=False,
                                                   init="fanintimesfanouttimestwo"))
model.add(keras.layers.core.DenseAfterRevcompWeightedSum(output_dim=10))
model.add(keras.layers.core.Activation("relu"))

model.add(keras.layers.core.Dense(output_dim=10))
model.add(keras.layers.core.Activation("sigmoid"))


model.compile(optimizer="sgd", loss="binary_crossentropy")

#randomly generate some inputs
rand_inp = np.random.random((10, 100, 4))

#confirm that forward and reverse-complement versions give same results
fwd_predict = model.predict(rand_inp)
rev_predict = model.predict(rand_inp[:, ::-1, ::-1])

#print the maximum value of the forward and reverse predictions
#should give 0.502919
print("Max prediction on forward seqs",np.max(fwd_predict))
print("Max prediction on revcomps",np.max(rev_predict))

#print the max difference in predictions
#should give 0.0
print("Maximum absolute difference:",np.max(np.abs(fwd_predict - rev_predict)))