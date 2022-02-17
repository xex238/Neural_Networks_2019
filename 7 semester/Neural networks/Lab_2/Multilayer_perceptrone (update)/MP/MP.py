import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import numpy as np
import tensorflow as tf
import keras as keras
import keras_metrics
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils

def scatterplot(x_data1, y_data1, color1, alpha1, x_data2, y_data2, color2, alpha2, graph_title, model):
    plt.title(graph_title)
    plt.scatter(x_data1, y_data1, color = color1, alpha = alpha1)   
    plt.scatter(x_data2, y_data2, color = color2, alpha = alpha2, marker = 's')
    eps = 0.1
    x = 0
    y = 0
    for i in range(200):
        x += eps
        y = 0
        for j in range(200):
            y += eps
            inputs = np.c_[1, x, y].astype(np.float)
            #print(inputs)
            outputs = model.predict_classes(inputs)
            #print(outputs)
            if (outputs == 0):
                plt.scatter(x, y, color = 'yellow', alpha = 0.1, s = 1)
            else:
                plt.scatter(x, y, color = 'green', alpha = 0.1, s = 5)
    plt.show()

def scatterplot_test(x_test, y_test, y_pred, color1, color2, alpha1, alpha2, graph_title):
    plt.title(graph_title)
    for i in range(x_test.shape[0]):
        if (y_pred[i] == 0) and (y_test[i, 0] == 1):
            plt.scatter(x_test[i, 1], x_test[i, 2], color='green', alpha=alpha1)
        elif (y_pred[i] == 1) and (y_test[i, 1] == 1):
            plt.scatter(x_test[i, 1], x_test[i, 2], color='yellow', alpha=alpha1)
        if (y_pred[i] == 0) and (y_test[i, 0] == 0):
            plt.scatter(x_test[i, 1], x_test[i, 2], color=color1, alpha=alpha2, marker = 's')
        if (y_pred[i] == 1) and (y_test[i, 1] == 0):
            plt.scatter(x_test[i, 1], x_test[i, 2], color=color2, alpha=alpha2, marker = 's')
    plt.show()

if __name__ == "__main__":
 path_1 = "dataset/data_banknote_authentication_Roma_dataset.txt"
 path_2 = "dataset/HTRU_2.csv"
 path_3 = "dataset/Iris.txt"
 file = pd.read_csv(path_3)

 nb_classes = 3
 model = Sequential()
 model.add(Dense(output_dim = 100, input_dim = 5, activation = 'relu')) # 'relu' - max(x, 0)
 model.add(Dense(output_dim = 200, activation = 'relu'))
 # softmax - обобщённая функция активации softsign для многомерного случая
 model.add(Dense(output_dim = nb_classes, activation = 'softmax'))
 model.summary()

 # shape возвращает количество строк и столбцов в матрице
 x0 = np.ones(file.shape[0]) # Единичный вход для нейронов
 inp = np.array(file.iloc[:, [0, 1, 2, 3]])
 inputs = np.c_[x0, inp].astype(np.float) # Добавляем единичный вход для нейронов
 outp = np.array(file.iloc[:, [4]])
 outputs = np.zeros((outp.shape[0], 3))

 print("inputs = ")
 print(inputs)
 print("outputs = ")
 print(outputs)

 for i in range(outp.shape[0]):
    if (outp[i, 0] == "Iris-setosa"):
        outputs[i, 0] = 1
    elif (outp[i, 0] == "Iris-versicolor"):
        outputs[i, 1] = 1
    elif (outp[i,0] == "Iris-virginica"):
        outputs[i, 2] = 1
    else:
        outputs[i,0] = -1

 print("inputs = ")
 print(inputs)
 print("outputs = ")
 print(outputs)

 #x_train, x_rest, y_train, y_rest = train_test_split(inputs, outputs, test_size = 0.3)
 #x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size = 0.5)

 #model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [keras_metrics.precision(), keras_metrics.recall(), 'accuracy', 'mse'])
 #history = model.fit(x_train, y_train, batch_size = 50, nb_epoch = 10, verbose = 0, validation_data = (x_val, y_val))
 #y_pred = model.predict_classes(x_test)

 #print('Accuracy =', history.history['accuracy'])
 #print('Precision =', history.history['precision'])
 #print('Recall =', history.history['recall'])
 #print('MSE =', history.history['mse'])

 #plt.plot(history.history['loss'])
 #plt.plot(history.history['val_loss'])
 #plt.title('Loss')
 #plt.ylabel('loss')
 #plt.xlabel('epoch')
 #plt.legend(['train', 'val'], loc = 'upper left')
 #plt.show()