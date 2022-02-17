import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU, Softmax, Activation, Input
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()
    

class MultipleLayerNeuralNet(tf.keras.Model):
    
    def __init__(self, neurons_counts):
        super(MultipleLayerNeuralNet, self).__init__()
        self.neurons = neurons_counts
        self.print_pattern = "loss {:10.4f} \t accuracy {:10.4f} \t MSE {:10.4f} \t precision {:10.4f} \t recall {:10.4f}"
        self.loss_values = []
        self.MSEs = []
        self.accuracys = []
        self.precisions = []
        self.recalls = []
        self.nn_stats = [self.loss_values, self.MSEs, self.accuracys, 
                         self.precisions, self.recalls]
    
    def initialize_layers(self, neurons):
        layers = []
        for i in range(1, len(neurons)):
            layers.append(Dense(neurons[i], input_shape=(neurons[i-1],)))
            layers.append(ReLU())
        layers.append(Softmax(axis=1))
        return layers
    
    def compute_metrics(self, x, y):
        preds = self.predict(x)        
        accuracy = (preds == y).sum() / len(preds)
        MSE = ((preds - y) ** 2).sum() / len(preds)
        TP = np.logical_and(preds == 1, preds == y).sum()
        FP = np.logical_and(preds == 1, preds != y).sum()
        FN = np.logical_and(preds == 0, preds != y).sum()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        return (accuracy, MSE, precision, recall)
    
    def batch_generator(self, x, y, batch_size):
        if batch_size == -1:
            return x, y
        n_batches = len(y) // batch_size
        if batch_size * n_batches < len(y):
            n_batches += 1
        for i in range(n_batches):
            yield x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]
            
    def extend_x(self, x):
        return np.concatenate((x, np.ones((len(x), 1))), axis=1)
    
    def gradient_step(self, x, y):
        y = y.astype(np.integer)
        with tf.GradientTape() as tape:
            preds = self.predict_proba(x)
            if self.n_outputs == 1:
                loss = self.one_output_loss(y, preds)
            else:
                loss = self.loss_function(y, preds)
        grads = tape.gradient(loss, self.trainable_variables)
        if self.train_mode == 'debug':
            print(tf.reduce_max(grads[0]))
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return tf.reduce_mean(loss)
    
    def init_decision_clusters(self, n_classes):
        clusters = []
        shift = 1 / n_classes
        for i in range(n_classes):
            clusters.append((shift * i + shift * (i+1)) / 2)
        return np.array(clusters)
    
    def one_output_loss(self, y, preds):
        right_centres = tf.constant([[self.clusters[int(i)]] for i in y.reshape((len(y),))])
        return (right_centres - preds) ** 2
    
    def fit(self, x, y, n_epochs=10, batch_size=-1, lr=0.001, train_mode='debug'):
        y = y.astype(np.integer)
        self.train_mode = train_mode
        self.neurons = [x.shape[1] + 1] + self.neurons
        self.n_outputs = self.neurons[-1]
        
        self.first_layer = Dense(self.neurons[1], 
                                 input_shape=(self.neurons[0],),
                                 kernel_initializer='glorot_normal')
        self.second_layer = Activation('relu')
        
        self.third_layer = Dense(self.neurons[2], 
                                 input_shape=(self.neurons[1],),
                                 kernel_initializer='glorot_normal')
        self.fourth_layer = Activation('relu')
        
        self.fifth_layer = Dense(self.neurons[3], 
                                 input_shape=(self.neurons[2],),
                                 kernel_initializer='glorot_normal')
        
        if self.n_outputs != 1:
            self.last_layer = Activation('softmax')
            self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
                                                        from_logits=False, reduction='none')
        else:
            self.last_layer = Activation('sigmoid')
            self.clusters = self.init_decision_clusters(len(np.unique(y)))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        print_every_n = 1
        for epoch in range(n_epochs):
            gen = self.batch_generator(x, y, batch_size)
            for batch_x, batch_y in gen:
                loss = self.gradient_step(batch_x, batch_y)
            accuracy, MSE, precision, recall = self.compute_metrics(x, y)
            self.loss_values.append(loss.numpy().sum())
            self.accuracys.append(accuracy)
            self.MSEs.append(MSE)
            self.precisions.append(precision)
            self.recalls.append(recall)
            if (epoch + 1) % print_every_n == 0 or epoch == 0:
                print('Epoch: {}'.format(epoch + 1))
                print(self.print_pattern.format(loss, accuracy, MSE, precision, recall))
    
    def predict(self, x):
        logits = self.predict_proba(x)
        if self.n_outputs == 1:
            result = []
            for log in logits:
                cl = np.argmin((self.clusters - log) ** 2)
                result.append(cl)
            return np.array(result)
        else:
            return tf.math.argmax(logits, axis=1).numpy()
    
    def predict_proba(self, x):
        x = self.extend_x(x)
        probs = self.first_layer(x)
        probs = self.second_layer(probs)
        probs = self.third_layer(probs)
        probs = self.fourth_layer(probs)
        probs = self.fifth_layer(probs)
        probs = self.last_layer(probs)
        return probs
