from numpy import random, abs, sqrt, sum, max, array
import numpy as np
from copy import deepcopy
from pandas import DataFrame
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, \
    fowlkes_mallows_score
from matplotlib import patches as patches
import matplotlib as mpl
import networkx as nx

class SOM:
    def __init__(self, net_x_dim, net_y_dim, num_features):
        self.network_dimensions = np.array([net_x_dim, net_y_dim])
        self.init_radius = min(self.network_dimensions[0], self.network_dimensions[1])
        # инициализируем вектор весов
        self.num_features = num_features
        self.initialize()

    def initialize(self):
        self.net = np.random.random((self.network_dimensions[0], self.network_dimensions[1], self.num_features))

    def train(self, data, num_epochs=100, init_learning_rate=0.01, resetWeights=False):
        if resetWeights:
            self.initialize()
        num_rows = data.shape[0]
        indices = np.arange(num_rows)
        self.time_constant = num_epochs / np.log(self.init_radius)
        for i in range(1, num_epochs + 1):
            # вычисляем новые значения α(t) and σ (t)
            radius = self.decay_radius(i)
            learning_rate = self.decay_learning_rate(init_learning_rate, i, num_epochs)
            # перемешиваем значения
            np.random.shuffle(indices)
            for record in indices:
                row_t = data[record, :]
                # находим ближайший нейрон
                bmu, bmu_idx = self.find_bmu(row_t)
                # обновляем веса
                for x in range(self.network_dimensions[0]):
                    for y in range(self.network_dimensions[1]):
                        weight = self.net[x, y, :].reshape(1, self.num_features)
                        w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
                        if w_dist <= radius ** 2:
                            influence = SOM.calculate_influence(w_dist, radius)
                            new_w = weight + (learning_rate * influence * (row_t - weight))
                            self.net[x, y, :] = new_w.reshape(1, self.num_features)

    def calculate_influence(distance, radius):
        return np.exp(-distance / (2 * (radius ** 2)))

    def find_bmu(self, row_t):
        bmu_idx = np.array([0, 0])
        min_dist = np.iinfo(np.int).max
        for x in range(self.network_dimensions[0]):
            for y in range(self.network_dimensions[1]):
                weight_k = self.net[x, y, :].reshape(1, self.num_features)
                sq_dist = np.sum((weight_k - row_t) ** 2)
                if sq_dist < min_dist:
                    min_dist = sq_dist
                    bmu_idx = np.array([x, y])
        bmu = self.net[bmu_idx[0], bmu_idx[1], :].reshape(1, self.num_features)
        return bmu, bmu_idx

    def predict(self, data):
        # find its Best Matching Unit
        bmu, bmu_idx = self.find_bmu(data)
        return bmu

    def decay_radius(self, iteration):
        return self.init_radius * np.exp(-iteration / self.time_constant)

    def decay_learning_rate(self, initial_learning_rate, iteration, num_iterations):
        return initial_learning_rate * np.exp(-iteration / num_iterations)

# Евклидова метрика
def EuclideanDistance(a, b):
    return sqrt(sum((a - b) ** 2))

# Метрика Чебышева
def ChebyshevDistance(a, b):
    return max(abs(a - b))

def K_means(X, number_of_clusters, metrica):
    k = number_of_clusters  # количество кластеров
    # случайным образом назначаем центры кластеров
    rng = np.random.RandomState(2)
    i = rng.permutation(X.shape[0])[:k]
    centers = X[i]
    # иницилизируем старые центры
    old_centers = np.zeros(centers.shape)
    # метки кластеров
    clusters = np.zeros(len(X))
    # расстояние между новыми и старыми центрами
    error = ChebyshevDistance(centers, old_centers)
    # повторяем алгоритм пока центры кластеров не будут изменяться
    while error != 0.0:
        # вычисляем расстояние между центрами кластеров и каждым объектом,
        # и объект приписывается к тому кластеру, к которому он ближе всего
        for i in range(len(X)):
            if (metrica == 'Euclidean'):
                distances = array([EuclideanDistance(X[i], centers[j]) for j in range(k)])
            if (metrica == 'Chebyshev'):
                distances = array([ChebyshevDistance(X[i], centers[j]) for j in range(k)])
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # запоминаем центры
        old_centers = deepcopy(centers)
        # находим новые центры, вычисляя средние значения для каждого кластера
        for i in range(k):
            points = array([X[j] for j in range(len(X)) if clusters[j] == i])
            centers[i] = points.mean(0)
        error = ChebyshevDistance(centers, old_centers)
    return clusters

def PlotShow(X, y, title):
    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    fig, ax = plt.subplots()
    ax.set_title(title)
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.show()

def Metrics(X, y_true, y_pred):
    print("Adjusted mutual info score = ", adjusted_mutual_info_score(y_true, y_pred, "arithmetic"))
    print("Adjusted rand score = ", adjusted_rand_score(y_true, y_pred))
    print("Calinski harabasz score = ", calinski_harabasz_score(X, y_pred))
    print("Fowlkes mallows score = ", fowlkes_mallows_score(y_true, y_pred))
    print()

def Drow_SOM(XX, y_pred_n, N, M):
    fig, ax = plt.subplots()
    ax.set_title('Карта Кохоннена')
    G = nx.grid_2d_graph(N, M)
    pos = {(i, j): (XX[i * M + j][0], XX[i * M + j][1]) for i, j in G.nodes()}
    labels = dict(((i, j), y_pred_n[i * M + j]) for i, j in G.nodes())
    cmap = plt.cm.Accent
    norm = mpl.colors.Normalize(vmin=min(y_pred_n), vmax=max(y_pred_n))
    m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    node_color = m.to_rgba(y_pred_n)
    node_color = array([node_color[i * M + j] for i, j in G.nodes()])
    nx.draw_networkx(G, pos=pos, labels=labels, node_color=node_color)
    plt.show()

if __name__ == "__main__":
    random.seed(3)
    number_clusters = 3
    number_features = 2
    centers = [[-700, 0], [0, 0], [0, 700]]
    X, y = make_blobs(n_samples=1000, centers=centers, n_features=2, cluster_std=0.7)
    PlotShow(X, y, 'Изначальный набор данных и принадлежность их к кластерам')

    y_pred_1 = K_means(X, number_clusters, 'Euclidean')
    PlotShow(X, y_pred_1, 'Кластеризация методом K-means, Eвклидова метрика')
    print('Оценки качества кластеризации методом K-means, Eвклидова метрика:')
    Metrics(X, y, y_pred_1)

    y_pred_2 = K_means(X, number_clusters, 'Chebyshev')
    PlotShow(X, y_pred_2, 'Кластеризация методом K-means, Чебышева метрика')
    print('Оценки качества кластеризации методом K-means, Чебышева метрика:')
    Metrics(X, y, y_pred_2)

    N = 5  # высота сетки
    M = 5  # длина сетки
    s = SOM(N, M, number_features)
    s.train(X)
    XX = array([s.net[i][j] for i in range(N) for j in range(M)])
    y_pred_n = K_means(XX, number_clusters, 'Euclidean')
    y_pred_som = np.zeros(len(X))
    for i in range(len(X)):
        win_neuron = s.predict(X[i])
        y_pred_som[i] = y_pred_n[np.where(XX == win_neuron)[0][0]]
    PlotShow(X, y_pred_som, 'Кластеризация с помощью SOM')
    print('Оценки качества кластеризации SOM:')
    Metrics(X, y, y_pred_som)
    Drow_SOM(XX, y_pred_n, N, M)


