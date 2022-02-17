import numpy as np
import random
import math
import pylab
from matplotlib import mlab
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def read_data():
    x=[]
    y=[]
    with open ("bank.txt", "r") as f:
        for ind,line in enumerate(f.readlines()):
            r=line.split(",")
            x.append([float(i) for l,i in enumerate(r) if 0<=l<=3])
            y.append(int(r[4]))
    return x,y
def vib():
    x,y=read_data()
    test_x=[]
    val_x=[]
    train_x=[]
    test_y=[]
    val_y=[]
    train_y=[]
    for i in range(len(x)):
        g = random.randint(0,9)
        if g==0:
            val_x.append(x[i])
            val_y.append(y[i])
            continue
        if g==1 or g==2:
            test_x.append(x[i])
            test_y.append(y[i])
            continue
        train_x.append(x[i])
        train_y.append(y[i])
    return(train_x,train_y,test_x,test_y,val_x,val_y)
tr_x,tr_y,tst_x,tst_y,val_x,val_y = vib()
print("тренировочная выборка:",len(tr_x))
print("тестовая выборка:",len(tst_x))
print("валидационная выборка:",len(val_x))

#Функция логистической ошибки
def loss(sig,y):
    return (-y*math.log(sig)-(1-y)*math.log(1-sig))
#Функция градиента
def grad(x, sig, y):
    d=x*(sig-y)
    return d
#Апдейт весов с шагом в сторону антиградиента
def update_ws(w, learn_rate, grad):
    return w-learn_rate*grad
#коэффициент обучения
kob=0.1
#Обучающая функция
def fit(x,y): 
    r_ws =np.array([0.5 for i in range(len(x[0]))]) #Веса
    for i in range(100000):
        k=random.randint(0,len(x)-1)
        npx=np.array(x[k])
        c_sig=sigmoid(np.dot(r_ws,npx))
        c_grad=grad(npx,c_sig,y[k])
        r_ws=update_ws(r_ws,kob,c_grad)
    return(r_ws)
def accuracy_metrics(y, ny):
    summ = 0
    y = np.array(y)
    ny = np.array(ny)
    for i in range(len(y)):
        if np.array_equal(y[i],ny[i]):
            summ += 1
    return summ / len(y)
def recall_metrics(y, ny):
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(y)):
        if y[i] == ny[i] and y[i] == 1:
            TP += 1
        elif y[i] == 1:
            FP += 1
        elif y[i] != ny[i] and y[i] == 0:
            FN += 1
    return TP / (TP + FN)
def precision_metrics(y, ny):
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(y)):
        if y[i] == ny[i]and y[i] == 1:
            TP += 1
        elif y[i]==1:
            FP += 1
    return TP / (TP + FP)
def predict(x,y,ws):
    lod=[]
    for i in range(len(x)):
        npx=np.array(x[i])
        c_sig=sigmoid(np.dot(npx,ws))
        if c_sig>0.5:
            lod.append(1)
        else:
            lod.append(0)
    print("accuracy:",accuracy_metrics(y,lod))
    print("recall:",recall_metrics(y,lod))
    print("precision:",precision_metrics(y,lod))
    for i in range(len(y)):
        print(lod[i],y[i])
ws=fit(tr_x,tr_y)
print(ws)
predict(val_x,val_y,ws)