import numpy as np
import tensorflow as tf

class SOMNetwork():
    def __init__(self, input_dim, dim=10, sigma=None, learning_rate=0.1, tay2=1000, dtype=tf.float32):
        #если сигма на определена устанавливаем ее равной половине размера решетки
        if not sigma:
            sigma = dim / 2
        self.dtype = dtype
        #определяем константы использующиеся при обучении
        self.dim = tf.constant(dim, dtype=tf.int64)
        self.learning_rate = tf.constant(learning_rate, dtype=dtype, name='learning_rate')
        self.sigma = tf.constant(sigma, dtype=dtype, name='sigma')
        #тау 1 (формула 6)
        self.tay1 = tf.constant(1000/np.log(sigma), dtype=dtype, name='tay1')
        #минимальное значение сигма на шаге 1000 (определяем по формуле 3)
        self.minsigma = tf.constant(sigma * np.exp(-1000/(1000/np.log(sigma))), dtype=dtype, name='min_sigma')
        self.tay2 = tf.constant(tay2, dtype=dtype, name='tay2')
        #input vector
        self.x = tf.placeholder(shape=[input_dim], dtype=dtype, name='input')
        #iteration number
        self.n = tf.placeholder(dtype=dtype, name='iteration')
        #матрица синаптических весов
        self.w = tf.Variable(tf.random_uniform([dim*dim, input_dim], minval=-1, maxval=1, dtype=dtype),
            dtype=dtype, name='weights')
        #матрица позиций всех нейронов, для определения латерального расстояния
        self.positions = tf.where(tf.fill([dim, dim], True))