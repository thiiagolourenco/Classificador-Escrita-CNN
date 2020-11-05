import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow
from keras.utils import np_utils

#Seed setada para reproduzir o experimento.
np.random.seed(2)

dataframe_train = pd.read_csv('train.csv')
dataframe_test = pd.read_csv('test.csv')

#Tiramos a infomação do digito
x_train = dataframe_train.drop(["label"], axis=1).values

#Informamos para o Keras a terceira dimensão, portanto: 28x28x1.
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = dataframe_test.values.reshape((dataframe_test.shape[0], 28, 28, 1))
"""Nesse ponto já tenho os dados X nas coordenadas e sem os rótulos."""

#One-hot-encoder da classe.
y_train = dataframe_train["label"].values
y_train = np_utils.to_categorical(y_train)

#Visualizando algumas imagens.
for i in range(0, 6):
    random_num = np.random.randint(0, len(x_train))
    img = x_train[random_num]
    plt.subplot(3, 2, i+1)
    plt.imshow(img.reshape(28, 28), cmap=plt.get_cmap('gray'))

plt.subplots_adjust(top=1.4)
plt.show()