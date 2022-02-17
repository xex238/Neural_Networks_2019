import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU, Softmax, Activation, Input
import numpy as np
import matplotlib.pyplot as plt

from model import MultipleLayerNeuralNet
tf.enable_eager_execution()


def generate_2d_point_cloud(n, E, D):
    return np.random.normal(E, D, (n, 2))

def data_gen(n_classes, n_samples_per_class, shift):
    arrays = []
    labels = []
    for i in range(n_classes):
        arrays.append(generate_2d_point_cloud(n_samples_per_class, 0, 1) + i * shift)
        labels += [[i]] * n_samples_per_class
    ls_data = np.concatenate(arrays, axis=0)
    ls_data = np.concatenate((ls_data, np.array(labels)), axis=1)
    np.random.shuffle(ls_data)
    return ls_data

if __name__ == '__main__':
    print('В данном запуске будет продемонстрированна работа модели на 3 сценариях: плохо разделимые данные с 2 классами, хорошо разделимые данные с 2 классами, идеально разделимые данные с 2 классами')
    n_classes = 2
    n_samples_per_class = 10000
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for j, shift in enumerate([2, 3, 5]):
        print('Shift: {}'.format(shift))
        ls_data = data_gen(n_classes, n_samples_per_class, shift)
        model = MultipleLayerNeuralNet([100, 50, n_classes])
        model.fit(ls_data[:, :-1], ls_data[:, -1], batch_size=1000, train_mode='release')
