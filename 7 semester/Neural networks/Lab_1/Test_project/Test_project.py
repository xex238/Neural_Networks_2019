import numpy as np

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# Входные данные
training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

print("Входные данные (training_inputs)")
print(training_inputs)

# Выходные данные
training_outputs = np.array([[0, 1, 1, 0]]).T

print("Выходные данные (training_outputs)")
print(training_outputs)

# Установка начального состояния для генератора случайных чисел
np.random.seed(1)

# Инициализация весов [-1; 1]
# random() - случайное число от 0 до 1
# (3, 1) - создание матрицы и запись в неё случайных чисел в промежутке [0; 1]
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Случайные инициализирующие веса (synaptic_weights)")
print(synaptic_weights)

# Метод обратного распространения
input_layer = training_inputs

# Функция dot возвращает скалярное произведение векторов (sigmoid - функция активации)
outputs = sigmoid(np.dot(input_layer, synaptic_weights))

print("Результат (outputs = sigmoid(np.dot(input_layer, synaptic_weights))):")
print(outputs)
    
# Поиск ошибки (разница между реальными выходными данными и тестовыми)
err = training_outputs - outputs

print("Ошибки (err = training_outputs - outputs):")
print(err)

# Adjustments - регулировка, корректировка
adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))

print("Корректировка (adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))):")
print(adjustments)

synaptic_weights += adjustments

print("Веса после одной итерации обучения (synaptic_weights += adjustments):")
print(synaptic_weights)