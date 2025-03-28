import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Activation,  Dropout,LSTM,Embedding,Layer, LSTM, Conv1D, MaxPool1D, TimeDistributed, Input, GRU, RNN
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def my_loss(y_true,y_pred):
    
    squared_difference = tf.square(y_true - y_pred)
    
    return tf.reduce_mean(squared_difference, axis=[1, 0])



inputs = Input((None, 5))

cell1 = tf.keras.layers.SimpleRNNCell(units=208)
cell2 = tf.keras.layers.SimpleRNNCell(units=208)
cell3 = tf.keras.layers.SimpleRNNCell(units=208)
cell4 = tf.keras.layers.SimpleRNNCell(units=208)



layer_1     = RNN(cell1, return_sequences=True)

layer_2     = RNN(cell2, return_sequences=True)

layer_3     = RNN(cell3, return_sequences=True)

layer_4     = RNN(cell4, return_sequences=True)

X           = layer_1(inputs)

X           = layer_2(X)

X           = layer_3(X)

V           = layer_4(X)


V  = Dense(240, activation='tanh')(V)

V  = Dense(64, activation='tanh')(V)

V1  = Dense(5, activation='tanh')(V)


model_RNN = Model(inputs=inputs, outputs=[V1])


opt = Adam(0.001)

model_RNN.compile(optimizer = opt, loss = my_loss)