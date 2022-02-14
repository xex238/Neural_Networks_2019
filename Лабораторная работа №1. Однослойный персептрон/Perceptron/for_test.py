#import functions
#import csv
#import numpy as np
#import random
#import math
#import copy

## Округление до большего целого
##math.ceil(x) 
## Округление до меньшего целого
##math.floor(x)

## Пример глубокого копирования (копирование по значению)
##result_A = [90, 85, 82]
##result_B = copy.deepcopy(result_A)

## Функция, меняющая значения массива в выходными значениями нейронов
#def change(key, discharge):
#    i = 0
#    if(key[i] + 1 == discharge):
#        while((key[i] + 1 == discharge) and (i + 1 < len(key))):
#            key[i] = 0
#            i = i + 1
#        if(i < len(key)):
#            key[i] = key[i] + 1
#            #print("key[i] = ", key[i])
#            #print("I do it!")
#    else:
#        key[i] = key[i] + 1
#    #print("key = ", key)
#    return key

#count_of_discharges = 3 # Количество разрядов
#count_of_neurons = 5
#count_of_signs = 2
#key = []
#key_value = []
#for i in range(count_of_neurons):
#    key.append(0)
#key[0] = -1

#print("key = ", key)

#class_value = 0
#for i in range(count_of_discharges ** count_of_neurons):
#    key = change(key, count_of_discharges)
#    print("key = ", key)

#    mas_helper = []
#    mas_helper.append(copy.deepcopy(key))
#    mas_helper.append(class_value)
#    key_value.append(mas_helper)

#    if (class_value + 1 == count_of_signs):
#        class_value = 0
#    else:
#        class_value = class_value + 1

#print(key_value)




from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples = 1000,n_features=2, centers=3,cluster_std = 1,center_box=(-8.0,8.0),shuffle=False)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

a = open('new_trial2.txt', 'w')
for i in range(X.shape[0]):
    string = str(X[i][0]) + ',' + str(X[i][1]) + ',' + str(y[i]) + '\n'
    a.write(string)
a.close()