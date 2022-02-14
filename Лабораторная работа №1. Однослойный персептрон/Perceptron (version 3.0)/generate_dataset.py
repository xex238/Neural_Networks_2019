from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples = 1000,n_features=6, centers=3,cluster_std = 0.1,center_box=(-2.0,2.0),shuffle=False)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

a = open('dataset/new_generated_dataset.txt', 'w')
for i in range(X.shape[0]):
    string = str(X[i][0]) + ',' + str(X[i][1]) + ',' + str(y[i]) + '\n'
    a.write(string)
a.close()