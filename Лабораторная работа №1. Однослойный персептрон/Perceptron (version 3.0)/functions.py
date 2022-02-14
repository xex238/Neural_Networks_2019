import numpy as np
import random
import matplotlib.pyplot as plt

# Конвертация значений
def converting(file_matrix):
    # Конвертируем признаки датасета
    try:
        for i in range(len(file_matrix)):
            for j in range(len(file_matrix[0]) - 1):
                file_matrix[i][j] = float(file_matrix[i][j])
    except:
        print("Плохие значения в датасете!")
        exit()

    # Конвертируем классы датасета
    try:
        for i in range(len(file_matrix)):
            file_matrix[i][len(file_matrix[0]) - 1] = float(file_matrix[i][len(file_matrix[0]) - 1])
    except ValueError:
        dict = {}
        counter = 0
        for i in range(len(file_matrix)):
            try:
                file_matrix[i][len(file_matrix[0]) - 1] = dict[file_matrix[i][len(file_matrix[0]) - 1]]
            except KeyError:
                dict[file_matrix[i][len(file_matrix[0]) - 1]] = counter
                file_matrix[i][len(file_matrix[0]) - 1] = counter
                counter = counter + 1
        print("Словарь для классов следующий:")
        print(dict)

# Функция для отрисовки массива точек на графике
def draw_points(points_matrix):
    fig = plt.figure()

    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(len(points_matrix)):
        if(points_matrix[2] == 0):
            x1.append(points_matrix[i][0])
            y1.append(points_matrix[i][1])
        if(points_matrix[2] == 1):
            x2.append(points_matrix[i][0])
            y2.append(points_matrix[i][1])

    figure_points = fig.add_subplot(111)
    figure_points.set_title("Точки для двумерных исходных данных")
    figure_points.scatter(x1, y1, color = 'green')
    figure_points.scatter(x2, y2, color = 'blue')

    plt.show()

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
def learning_function(inputs, outputs, weights, speed, activation_function, MSE_mas, accuracy_mas, count_of_neurons):
    # MSE - Mean Squared Error (среднеквадратичная ошибка)
    MSE = 0

    TP = 0 # True Positive (Правильно определена 1)
    FP = 0 # False Positive (Неправильно определена 1)
    FN = 0 # False Negative (Неправильно определён 0)
    TN = 0 # True Negative (Правильно определён 0)

    for j in range(len(inputs)):
        output = []
        err = []
        for i in range(count_of_neurons): # len(synaptic_weights = кол-во столбцов входных данных)
            #count_of_neurons = count_of_signs

            value_synaptic_weights = []
            for k in range(len(weights)):
                value_synaptic_weights.append(weights[k][i])

            sum = scalar(inputs[j], value_synaptic_weights)
            if(activation_function == "softsign"):
                output.append(softsign_activation_function(sum))
            if(activation_function == "sigmoid"):
                output.append(sigmoid_activation_function(sum))
            if(i == outputs[j]):
                err.append(1 - output[i])
            else:
                err.append(-output[i])
            MSE = MSE + (err[i] * err[i])
        
        #count_of_repeats = 0
        #for i in range(count_of_neurons): # len(synaptic_weights = count_of_signs)
        #    if(round(output[i]) == 1):
        #        count_of_repeats = count_of_repeats + 1
        
        #if(count_of_repeats == 1):
        #    for i in range(count_of_neurons): # len(synaptic_weights = count_of_signs)
        #        if((round(output[i]) == 1) and (i == outputs[j])):
        #            TP = TP + 1
        #        if((round(output[i]) == 1) and (i != outputs[j])):
        #            FN = FN + 1
        #else:
        #    FP = FP + 1

        for i in range(count_of_neurons):
            if((round(output[i]) == 0) and (i != outputs[j])):
                TN = TN + 1
            if((round(output[i]) == 0) and (i == outputs[j])):
                FN = FN + 1
            if((round(output[i]) == 1) and (i == outputs[j])):
                TP = TP + 1
            if((round(output[i]) == 1) and (i != outputs[j])):
                FP = FP + 1

        #if(j < 10):
        #    print("sum = ", sum)
        #    print("learn_training_outputs[", j, "] = ", learn_training_outputs[j])
        #    print("outputs = ", output)
        #    print("err = ", err)

        if(activation_function == "softsign"):
            for i in range(len(weights)):
                for k in range(len(weights[0])):
                    weights[i][k] = weights[i][k] + speed * err[k] * derivative_softsign_activation_function(sum) * inputs[j][i]
                    #weights[i][k] = weights[i][k] + speed * err[k] * inputs[j][i]
        if(activation_function == "sigmoid"):
            for i in range(len(weights)):
                for k in range(len(weights[0])):
                    weights[i][k] = weights[i][k] + speed * err[k] * derivative_sigmoid_activation_function(sum) * inputs[j][i]
                    #weights[i][k] = weights[i][k] + speed * err[k] * inputs[j][i]
    MSE = MSE / len(inputs)
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
def test_function(inputs, outputs, weights, activation_function, MSE_mas, accuracy_mas, count_of_neurons, marker):
    # MSE - Mean Squared Error (среднеквадратичная ошибка)
    MSE = 0

    TP = 0 # True Positive (Правильно определена 1)
    FP = 0 # False Positive (Неправильно определена 1)
    FN = 0 # False Negative (Неправильно определён 0)
    TN = 0 # True Negative (Правильно определён 0)

    for j in range(len(inputs)):
        output = []
        err = []
        for i in range(count_of_neurons): # len(synaptic_weights = кол-во столбцов входных данных)
            #count_of_neurons = count_of_signs

            value_synaptic_weights = []
            for k in range(len(weights)):
                value_synaptic_weights.append(weights[k][i])

            sum = scalar(inputs[j], value_synaptic_weights)
            if(activation_function == "softsign"):
                output.append(softsign_activation_function(sum))
            if(activation_function == "sigmoid"):
                output.append(sigmoid_activation_function(sum))
            if(i == outputs[j]):
                err.append(1 - output[i])
            else:
                err.append(-output[i])
            MSE = MSE + (err[i] * err[i])
        
        #count_of_repeats = 0
        #for i in range(count_of_neurons): # len(synaptic_weights = count_of_signs)
        #    if(round(output[i]) == 1):
        #        count_of_repeats = count_of_repeats + 1
        
        #if(count_of_repeats == 1):
        #    for i in range(count_of_neurons): # len(synaptic_weights = count_of_signs)
        #        if((round(output[i]) == 1) and (i == outputs[j])):
        #            TP = TP + 1
        #        if((round(output[i]) == 1) and (i != outputs[j])):
        #            FN = FN + 1
        #else:
        #    FP = FP + 1

        for i in range(count_of_neurons):
            if((round(output[i]) == 0) and (i != outputs[j])):
                TN = TN + 1
            if((round(output[i]) == 0) and (i == outputs[j])):
                FN = FN + 1
            if((round(output[i]) == 1) and (i == outputs[j])):
                TP = TP + 1
            if((round(output[i]) == 1) and (i != outputs[j])):
                FP = FP + 1

        #if(j < 10):
        #    print("sum = ", sum)
        #    print("learn_training_outputs[", j, "] = ", learn_training_outputs[j])
        #    print("outputs = ", output)
        #    print("err = ", err)

        if(marker == 1):
            if(i < 100):
                print("outputs = ", output)
                print("test_training_outputs[", i, "] = ", outputs[i])
                print()

    MSE = MSE / len(inputs)
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

# Проверка датасета на пригодность к использованию
def check_dataset(file_matrix):
    for i in range(len(file_matrix)):
        #print("len(file_matrix[i]) = ", len(file_matrix[i]))
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