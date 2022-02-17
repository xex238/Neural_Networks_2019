# Комментирование нескольких строк кода Ctrl + K, C
# Раскомментирование нескольких строк кода Ctrl + K, U
# Сдвиг строк кода вправо Tab
# Сдвиг строк кода влево Shift + Tab

import numpy as np

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def sigmoid_2(x):
    return (np.exp(-x) / ((1 + np.exp(-x)) * (1 + np.exp(-x))))

# Входные данные (первоначальные)
#training_inputs = np.array([[0, 0, 1],
#                            [1, 1, 1],
#                            [1, 0, 1],
#                            [0, 1, 1]])
training_inputs = np.array([[0, 1, 0],
                            [0, 1, 1],
                            [0, 0, 1],
                            [1, 1, 1]])

print("Входные данные")
print(training_inputs)

# Выходные данные (первоначальные)
#training_outputs = np.array([[0, 1, 1, 0]]).T
training_outputs = np.array([[1, 1, 0, 1]]).T
print("Выходные данные")
print(training_outputs)

# Введение скорости обучения персептрона
print("Введите скорость обучения персептрона")
speed = float(input())

# Установка начального состояния для генератора случайных чисел
np.random.seed(1)

# Инициализация весов [-1; 1]
# random() - случайное число от 0 до 1
# (3, 1) - создание матрицы и запись в неё случайных чисел в промежутке [0; 1]
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Случайные инициализирующие веса")
print(synaptic_weights)


# Начало тестового кода
for i in range(4):
    print("training_inputs[", i, "]")
    print(training_inputs[i])
# Конец тестового кода


# Начало тестирующего кода

#sum = np.dot(training_inputs, synaptic_weights)
#print("Сумма равна, sum =")
#print(sum)

#outputs = sigmoid(sum)
#print("Значения после функции активации, outputs = sigmoid(sum) =")
#print(outputs)

#err = training_outputs - outputs
#print("Ошибка равна, err = training_outputs - outputs = ")
#print(err)

#adjustments = np.dot(training_inputs.T, err * (outputs * (1 - outputs)))
#print("Поправка весов, adjustments = ")
#print(adjustments)

#synaptic_weights += adjustments
#print("Новые веса, synaptic_weights = ")
#print(synaptic_weights)

# Конец тестирующего кода

# Метод обратного распространения (по всем значениям)
print()
for i in range(20000):
    # Функция dot возвращает скалярное произведение векторов
    sum = np.dot(training_inputs, synaptic_weights)
    # Функция активации
    outputs = sigmoid(sum)
    
    # Поиск ошибки (разница между реальными выходными данными и тестовыми)
    err = training_outputs - outputs
    # Adjustments - регулировка, корректировка
    adjustments = np.dot(training_inputs.T, err * (outputs * (1 - outputs)))

    synaptic_weights += adjustments

print("Веса после обучения №1:")
print(synaptic_weights)

print("Результат №1:")
print(outputs)

# "Обнуление" весов
np.random.seed(1)
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Случайные инициализирующие веса №2")
print(synaptic_weights)

# Метод обратного распространения (по одному значению)
for i in range(5000):
    for j in range(4):
        sum = np.dot(training_inputs[j], synaptic_weights)
        outputs = sigmoid(sum)
        err = training_outputs[j] - outputs

        for w in range(3):
            synaptic_weights[w] = synaptic_weights[w] + speed * err * sigmoid_2(sum) * training_inputs[j][w]

print("Веса после обучения №2:")
print(synaptic_weights)

print("Результат №2:")
print(outputs)

# Тест
new_inputs = np.array([[0, 0, 1]])
new_inputs_1 = np.array([[0, 0, 0],
[0, 0, 1],
[0, 1, 0],
[0, 1, 1],
[1, 0, 0],
[1, 0, 1],
[1, 1, 0],
[1, 1, 1]])
for i in range(8):
    output = sigmoid(np.dot(new_inputs_1[i], synaptic_weights))
    print(new_inputs_1[i], " ", output)
#output = sigmoid(np.dot(new_inputs, synaptic_weights))

#print("Новая ситуация")
#print(output)