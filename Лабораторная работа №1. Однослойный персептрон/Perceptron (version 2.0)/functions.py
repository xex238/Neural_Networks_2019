import numpy as np
import random
import matplotlib.pyplot as plt

# Функция, меняющая значения массива в выходными значениями нейронов
def change(key, discharge):
    i = 0
    if(key[i] + 1 == discharge):
        while((key[i] + 1 == discharge) and (i + 1 < len(key))):
            key[i] = 0
            i = i + 1
        if(i < len(key)):
            key[i] = key[i] + 1
            #print("key[i] = ", key[i])
            #print("I do it!")
    else:
        key[i] = key[i] + 1
    #print("key = ", key)
    return key

# Функция для отрисовки графиков
def graph_drawing_function(MSE_mas_learning, MSE_mas_test, accuracy_mas_learning, accuracy_mas_test):
    # Массив для значений x
    helper_mas = []
    for i in range(len(MSE_mas_learning)):
        helper_mas.append(i)

    fig = plt.figure()
    MSE_graph = fig.add_subplot(1, 2, 1)
    MSE_graph.set_title("green - MSE for learning, blue - MSE for validation")
    MSE_graph.scatter(helper_mas, MSE_mas_learning, color = 'green', marker = '*')
    MSE_graph.scatter(helper_mas, MSE_mas_test, color = 'blue', marker = '*')
    MSE_graph.plot(helper_mas, MSE_mas_learning, color = 'green')
    MSE_graph.plot(helper_mas, MSE_mas_test, color = 'blue')

    accuracy_graph = fig.add_subplot(1, 2, 2)
    accuracy_graph.set_title("green - accuracy for learning, blue - accuracy for validation")
    accuracy_graph.scatter(helper_mas, accuracy_mas_learning, color = 'green', marker = '*')
    accuracy_graph.scatter(helper_mas, accuracy_mas_test, color = 'blue', marker = '*')
    accuracy_graph.plot(helper_mas, accuracy_mas_learning, color = 'green')
    accuracy_graph.plot(helper_mas, accuracy_mas_test, color = 'blue')

    plt.show()

# Считывание с csv файла
def read_from_csv_file(path):
    file_open = open(csv_path, "r")
    file_reader = csv.reader(file_open)
    file_matrix = []
    for row in file_reader:
        file_matrix.append(row)
    return file_matrix

# Считывание с файла (любого)
def read_from_file(path):
    file_reader = open(path)
    all_file = file_reader.read()
    file_mas = []
    file_mas = all_file.split("\n")
    file_matrix = []
    for i in file_mas:
        file_matrix.append(i.split(","))
    return file_matrix

# Функция для определения класса на основе выходного значения обучающей выборки
def choose_sign(output, signs):
    min = abs(signs[0] - output)
    i_min = 0
    for i in range(len(signs)):
        if(abs(signs[i] - output) < min):
            min = abs(signs[i] - output)
            i_min = i
    class_output = signs[i_min]
    return class_output

# Функция обучения на заданной выборке
def learning_function(training_inputs, training_outputs, synaptic_weights, speed, signs, activation_function, MSE_mas, accuracy_mas):
    # MSE - Mean Squared Error (среднеквадратичная ошибка)
    MSE = 0

    TP = 0 # True Positive (Правильно определена 1)
    FP = 0 # False Positive (Неправильно определена 1)
    FN = 0 # False Negative (Неправильно определён 0)
    TN = 0 # True Negative (Правильно определён 0)
    for j in range(len(training_inputs)):
        err = []
        for i in range(len(synaptic_weights)):
            sum = scalar(training_inputs[j], synaptic_weights[i])
            if(activation_function == "softsign"):
                output = softsign_activation_function(sum)
            if(activation_function == "sigmoid"):
                output = sigmoid_activation_function(sum)
            err = training_outputs[j] - output
            MSE = MSE + (err * err)

            class_output = choose_sign(output, signs)

        # Матрица ошибок для двух классов
        if(len(signs) == 2):
            if(class_output == training_outputs[j]):
                if(class_output == signs[0]):
                    TN = TN + 1
                if(class_output == signs[1]):
                    TP = TP + 1
            else:
                if(class_output == signs[0]):
                    FN = FN + 1
                if(class_output == signs[1]):
                    FP = FP + 1

        #err = err * derivative_softsign_activation_function(sum)
        #if(j < 10):
        #    print("sum = ", sum)
        #    print("learn_training_outputs[", j, "] = ", learn_training_outputs[j])
        #    print("outputs = ", output)
        #    print("err = ", err)

        if(activation_function == "softsign"):
            for w in range(len(synaptic_weights)):
                #synaptic_weights[w] = synaptic_weights[w] + speed * err * derivative_softsign_activation_function(sum) * training_inputs[j][w]
                synaptic_weights[w] = synaptic_weights[w] + speed * err * training_inputs[j][w]
        if(activation_function == "sigmoid"):
            for w in range(len(synaptic_weights)):
                #synaptic_weights[w] = synaptic_weights[w] + speed * err * derivative_sigmoid_activation_function(sum) * training_inputs[j][w]
                synaptic_weights[w] = synaptic_weights[w] + speed * err * training_inputs[j][w]
    MSE = MSE / len(training_inputs)
    MSE_mas.append(MSE)
    print("Среднеквадратичная ошибка в данной эпохе составила", MSE)
    print("TP = ", TP)
    print("FP = ", FP)
    print("FN = ", FN)
    print("TN = ", TN)
    # Точность работы алгоритма
    accuracy_mas.append((TP + TN) / (TP + TN + FP + FN))
    print("accurancy = ", (TP + TN) / (TP + TN + FP + FN))
    print("precision = ", TP / (TP + FP))
    print("recall = ", TP / (TP + FN))

# Функция проверки результатов на заданной выборке
def test_function(test_inputs, test_outputs, synaptic_weights, signs, marker, MSE_mas, accuracy_mas):
    MSE = 0
    TP = 0 # True Positive (Правильно определена 1)
    FP = 0 # False Positive (Неправильно определена 1)
    FN = 0 # False Negative (Неправильно определён 0)
    TN = 0 # True Negative (Правильно определён 0)
    for i in range(len(test_inputs)):
        sum = scalar(test_inputs[i], synaptic_weights)
        output = softsign_activation_function(sum)
        err = test_outputs[i] - output
        MSE = MSE + (err * err)

        class_output = choose_sign(output, signs)

        # Матрица ошибок для двух классов
        if(len(signs) == 2):
            if(class_output == test_outputs[i]):
                if(class_output == signs[0]):
                    TN = TN + 1
                if(class_output == signs[1]):
                    TP = TP + 1
            else:
                if(class_output == signs[0]):
                    FN = FN + 1
                if(class_output == signs[1]):
                    FP = FP + 1
        if(marker == 1):
            if(i < 100):
                print("outputs = ", output)
                print("test_training_outputs[", i, "] = ", test_outputs[i])
                print()
    MSE = MSE / len(test_inputs)
    MSE_mas.append(MSE)
    print("Среднеквадратичная ошибка в данной эпохе составила", MSE)
    print("TP = ", TP)
    print("FP = ", FP)
    print("FN = ", FN)
    print("TN = ", TN)
    # Точность работы алгоритма
    accuracy_mas.append((TP + TN) / (TP + TN + FP + FN))
    print("accuracy = ", (TP + TN) / (TP + TN + FP + FN))
    print("precision = ", TP / (TP + FP))
    print("recall = ", TP / (TP + FN))
    return (TP + TN) / (TP + TN + FP + FN)

# Проверка датасета на пригодность к использованию
def check_dataset(file_matrix):
    for i in range(len(file_matrix)):
        if(len(file_matrix[0]) != len(file_matrix[i])):
            print("Датасет испорчен. Количество столбцов различное")
            exit(0)

# Линейная нормализация матрицы
def linear_matrix_normalization(file_matrix):
    for i in range(len(file_matrix[0])):
        max = file_matrix[0][i]
        min = file_matrix[0][i]
        for j in range(len(file_matrix)):
            if(file_matrix[j][i] > max):
                max = file_matrix[j][i]
            if(file_matrix[j][i] < min):
                min = file_matrix[j][i]
        for j in range(len(file_matrix)):
            file_matrix[j][i] = (file_matrix[j][i] - min) / (max - min)
    #print("Нормализованная исходная матрица")
    #print(file_matrix[0])
    return file_matrix

# Линейная нормализация массива
def linear_mas_normalization(file_mas):
    max = file_mas[0]
    min = file_mas[0]
    for i in range(len(file_mas)):
        if(file_mas[i] > max):
            max = file_mas[i]
        if(file_mas[i] < min):
            min = file_mas[i]
    for i in range(len(file_mas)):
        file_mas[i] = (file_mas[i] - min) / (max - min)
    #print("Нормализованный исходный массив")
    #print(file_mas)
    return file_mas

# Нелинейная нормализация матрицы
def not_linear_matrix_normalization(file_matrix):
    a = 0.5 # Коэффициент нормализации
    for i in range(len(file_matrix[0])):
        average = 0
        for j in range(len(file_matrix)):
            average = average + file_matrix[j][i]
        average = average / len(file_matrix)
        for j in range(len(file_matrix)):
            file_matrix[j][i] = 1 / (np.exp((-1) * a * (file_matrix[j][i] - average)) + 1)
    #print("Нормализованная исходная матрица")
    #print(file_matrix[0])
    return file_matrix

# Скалярное произведение массивов
def scalar(mas_1, mas_2):
    result_mas = 0
    if((len(mas_1) == 0) | (len(mas_2) == 0)):
        print("Один из массивов пуст! Перемножать нечего!")
    elif(type(mas_1) != type(mas_2)):
        print("Массивы разных типов! Нельзя найти их скалярное произведение!")
        print("Тип массива 1 = ", type(mas_1))
        print("Тип массива 2 = ", type(mas_2))
    elif(len(mas_1) != len(mas_2)):
        print("Массивы разных размеров. Нельзя найти их скалярное произведение!")
    else:
        for i in range(len(mas_1)):
            result_mas += mas_1[i] * mas_2[i]
        return result_mas

# Функция активации (softsign)
def softsign_activation_function(x):
    return (x / (1 + np.abs(x)))

# Производная функции активации (softsign)
def derivative_softsign_activation_function(x):
    if(x >= 0):
        return (1 / ((1 + x) * (1 + x)))
    else:
        return (1 / ((1 - x) * (1 - x)))

# Функция активации (sigmoid)
def sigmoid_activation_function(x):
    return (1 / (1 + np.exp(-x)))

# Производная функция активации (sigmoid)
def derivative_sigmoid_activation_function(x):
    return (np.exp(-x) / ((1 + np.exp(-x)) * (1 + np.exp(-x))))