import numpy as np
import random
import matplotlib.pyplot as plt
import copy

class multilayer_perceptrone:
    count_of_layers = 0 # Количество общих слоёв в персептроне (входной + скрытые + выходной)
    count_of_neurons = [] # Количество нейронов в каждом слое
    neurons = [] # Матрица (список списков) со значениями нейронов
    delta_neurons = [] # Матрица (список списков) с пересчитанными значениями нейронов
    sum = [] # Для каждого нейрона сумма скалярное произведение значений входящих в него нейронов на веса
    
    synaptic_weights = [] # Трёхмерный массив весов. [слой][откуда выходит][куда входит]

    file_matrix = [] # Переменная для хранения исходного датасета

    classes = [] # Классы в исходном датасете
    count_of_classes = 0 # Количество классов в исходном датасете
    count_of_signs = 0 # Количество признаков в исходном датасете

    speed = 0 # Скорость обучения персептрона
    count_of_eras = 0 # Количество эпох обучения

    learn_inputs = [] # Обучающие входные данные
    learn_outputs = [] # Обучающие выходные данные
    test_inputs = [] # Тестовые входные данные
    test_outputs = [] # Тестовые выходные данные
    validation_inputs = [] # Валидационные входные данные
    validation_outputs = [] # Валидационные выходные данные

    test_probability = 0 # Количество % из всего датасета для тестового множества
    validation_probability = 0 # Количество % из всего датасета для валидационного множества

    # Добавить возможность сделать различную функцию активации для каждого слоя
    activation_function = [] # Функция активации для каждого слоя

    MSE_mas_learning = [] # Среднеквадратичная ошибка для обучающей выборки
    MSE_mas_test = [] # Среднеквадратичная ошибка для тестовой выборки
    MSE_validation = 0 # Среднеквадратичная ошибка для валидационной выборки
    accuracy_mas_learning = [] # Значение accuracy для обучающей выборки
    accuracy_mas_test = [] # Значение accuracy для тестовой выборки
    accuracy_validation = 0 # Значение accuracy для валидационной выборки

    min_MSE = -1 # Минимальная среднеквадратичная ошибка для тестовой выборки
    min_MSE_era = 0 # Значение эпохи для минимальной среднеквадратичной ошибки для тестовой выборки
    best_synaptic_weights = [] # Значения весов для минимальной среднеквадратичной ошибки

    # Конструктор класса
    def __init__(self, dataset_path, count_of_layers, test_probability, validation_probability, activation_function, neurons_hidden, count_of_eras, speed):
        self.read_from_file(dataset_path)

        self.count_of_layers = count_of_layers
        if(count_of_layers != len(neurons_hidden) + 2):
            print("Неверно определено количество слоёв и количество нейронов в слоях")
            exit()
        self.count_of_neurons = neurons_hidden
        self.count_of_eras = count_of_eras

        if(type(activation_function) == str):
            for i in range(count_of_layers):
                self.activation_function.append(activation_function)
        elif(type(activation_function) == list):
            if(len(activation_function) == count_of_layers):
                self.activation_function = activation_function
            else:
                print("Количество значений в функции активации не соответствует количеству слоёв")
                exit()
        else:
            print("Неопознанная функция активации")
            exit()

        self.test_probability = test_probability
        self.validation_probability = validation_probability

        self.speed = speed

    # Конвертация значений в тип float. Создание словаря для классов, если классы не цифры
    def converting(self):
        # Конвертируем признаки датасета
        try:
            for i in range(len(self.file_matrix)):
                for j in range(len(self.file_matrix[0]) - 1):
                    self.file_matrix[i][j] = float(self.file_matrix[i][j])
        except:
            print("Плохие значения в датасете!")
            exit()

        # Конвертируем классы датасета
        try:
            for i in range(len(self.file_matrix)):
                self.file_matrix[i][len(self.file_matrix[0]) - 1] = float(self.file_matrix[i][len(self.file_matrix[0]) - 1])
        except ValueError:
            dict = {}
            counter = 0
            for i in range(len(self.file_matrix)):
                try:
                    self.file_matrix[i][len(self.file_matrix[0]) - 1] = dict[self.file_matrix[i][len(self.file_matrix[0]) - 1]]
                except KeyError:
                    dict[self.file_matrix[i][len(self.file_matrix[0]) - 1]] = counter
                    self.file_matrix[i][len(self.file_matrix[0]) - 1] = counter
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
    def graph_drawing_function(self):
        # Массив для значений x
        helper_mas = []
        for i in range(len(self.MSE_mas_learning)):
            helper_mas.append(i)

        fig = plt.figure()
        MSE_graph = fig.add_subplot(1, 2, 1)
        MSE_graph.set_title("green - MSE for learning, blue - MSE for validation")
        MSE_graph.scatter(helper_mas, self.MSE_mas_learning, color = 'green', marker = '*')
        MSE_graph.scatter(helper_mas, self.MSE_mas_test, color = 'blue', marker = '*')
        MSE_graph.plot(helper_mas, self.MSE_mas_learning, color = 'green')
        MSE_graph.plot(helper_mas, self.MSE_mas_test, color = 'blue')

        accuracy_graph = fig.add_subplot(1, 2, 2)
        accuracy_graph.set_title("green - accuracy for learning, blue - accuracy for validation")
        accuracy_graph.scatter(helper_mas, self.accuracy_mas_learning, color = 'green', marker = '*')
        accuracy_graph.scatter(helper_mas, self.accuracy_mas_test, color = 'blue', marker = '*')
        accuracy_graph.plot(helper_mas, self.accuracy_mas_learning, color = 'green')
        accuracy_graph.plot(helper_mas, self.accuracy_mas_test, color = 'blue')

        plt.show()

    # Считывание с csv файла
    def read_from_csv_file(self, path):
        file_open = open(csv_path, "r")
        file_reader = csv.reader(file_open)
        for row in file_reader:
            self.file_matrix.append(row)

    # Считывание с файла (любого)
    def read_from_file(self, path):
        file_reader = open(path)
        all_file = file_reader.read()
        file_mas = []
        file_mas = all_file.split("\n")
        for i in file_mas:
            self.file_matrix.append(i.split(","))

    # Единичный вход для нейронов
    def single_inputs_for_neurons(self):
        for i in range(len(self.learn_inputs)):
            self.learn_inputs[i].append(1)
        for i in range(len(self.test_inputs)):
            self.test_inputs[i].append(1)
        for i in range(len(self.validation_inputs)):
            self.validation_inputs[i].append(1)

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

    # Функция для определения количества классов, количества нейронов и количества весов
    # Количество нейронов в выходном слое равно количеству весов
    def get_classes(self):
        self.count_of_signs = len(self.file_matrix[0]) - 1
        print("Количество признаков равно ", self.count_of_signs)        

        self.classes.append(self.file_matrix[0][len(self.file_matrix[0]) - 1])
        for i in range(len(self.file_matrix)):
            find = 0
            for j in range(len(self.classes)):
                if(self.classes[j] == self.file_matrix[i][len(self.file_matrix[i]) - 1]):
                    find = 1
            if(find == 0):
                self.classes.append(self.file_matrix[i][len(self.file_matrix[i]) - 1])

        self.count_of_classes = len(self.classes)
        print("Количество классов равно", self.count_of_classes)
        print(self.classes)

        print("Количество нейронов в каждом слое.")
        self.count_of_neurons.insert(0, self.count_of_signs)
        self.count_of_neurons.append(self.count_of_classes)

        for i in range(self.count_of_layers):
            print("Слой № ", i + 1, ". Количество нейронов - ", self.count_of_neurons[i])

        # Количество нейронов в первом слое равно количеству признаков
        self.count_of_neurons[0] = self.count_of_signs
        # Количество нейронов в последнем слое равно количеству классов
        self.count_of_neurons[len(self.count_of_neurons) - 1] = self.count_of_classes

        for i in range(self.count_of_layers):
            helper_mas = []
            for j in range(self.count_of_neurons[i]):
                helper_mas.append(0)
            self.neurons.append(copy.deepcopy(helper_mas))
            self.delta_neurons.append(copy.deepcopy(helper_mas))
            self.sum.append(copy.deepcopy(helper_mas))

    # Функция генерации случайных значений для весов
    def get_random_synaptic_weights(self):
        np.random.seed(1)

        for k in range(self.count_of_layers - 1):
            helper_mas_1 = []

            for i in range(self.count_of_neurons[k]):
                helper_mas_2 = []
                for j in range(self.count_of_neurons[k + 1]):
                    helper_mas_2.append(2 * np.random.random() - 1)
                helper_mas_1.append(helper_mas_2)
            self.synaptic_weights.append(helper_mas_1)

        #print("Случайные инициализирующие веса")
        #print(self.synaptic_weights)

    # Функция для разделения исходного датасета на множества случайным способом
    def devide_learning_test_validation(self):
        for i in range(len(self.file_matrix)):
            random_number = random.randint(1, 100)
            if(random_number > self.test_probability + self.validation_probability):
                self.learn_inputs.append(self.file_matrix[i])
                self.learn_outputs.append(self.file_matrix[i][len(self.file_matrix[i]) - 1])
                self.learn_inputs[len(self.learn_inputs) - 1].pop()
            elif(random_number <= self.validation_probability):
                self.validation_inputs.append(self.file_matrix[i])
                self.validation_outputs.append(self.file_matrix[i][len(self.file_matrix[i]) - 1])
                self.validation_inputs[len(self.validation_inputs) - 1].pop()
            else:
                self.test_inputs.append(self.file_matrix[i])
                self.test_outputs.append(self.file_matrix[i][len(self.file_matrix[i]) - 1])
                self.test_inputs[len(self.test_inputs) - 1].pop()

        print("Размер обучающей выборки равен:", len(self.learn_inputs))
        print("Это составляет", round((len(self.learn_inputs) / len(self.file_matrix) * 100), 2), "% от исходной выборки")
        print("Размер тестовой выборки равен:", len(self.test_inputs))
        print("Это составляет", round((len(self.test_inputs) / len(self.file_matrix) * 100), 2), "% от исходной выборки")
        print("Размер валидационной выборки равен:", len(self.validation_inputs))
        print("Это составляет", round((len(self.validation_inputs) / len(self.file_matrix) * 100), 2), "% от исходной выборки")
        print()

    # Функция обучения персептрона
    def learning_procedure(self):
        counter = 0 # Счётчик эпох

        # Процесс обучения
        while(counter < self.count_of_eras):
            print("Эпоха обучения №", counter + 1)
            print("Сейчас идёт обучающая выборка")
            self.learning_function()
            print("Сейчас идёт тестовая выборка")
            self.test_function()
            counter = counter + 1
            if((self.min_MSE == -1) or (self.MSE_mas_test[len(self.MSE_mas_test) - 1] < self.min_MSE)):
                self.min_MSE = self.MSE_mas_test[len(self.MSE_mas_test) - 1]
                self.min_MSE_era = counter
                self.best_synaptic_weights = self.synaptic_weights

        # Рисуем графики для MSE
        self.graph_drawing_function()

        # Проверка работы нейронной сети на валидационной выборке
        self.validation_function()

    # Функция обучения на заданной выборке
    def learning_function(self):
        # MSE - Mean Squared Error (среднеквадратичная ошибка)
        MSE = 0

        TP = 0 # True Positive (Правильно определена 1)
        FP = 0 # False Positive (Неправильно определена 1)
        FN = 0 # False Negative (Неправильно определён 0)
        TN = 0 # True Negative (Правильно определён 0)

        for i in range(len(self.learn_inputs)):
            # Высчитываем значения нейронов
            for j in range(self.count_of_layers):
                if(j == 0):
                    for k in range(self.count_of_signs):
                        self.neurons[0][k] = self.learn_inputs[i][k]
                else:
                    for k in range(self.count_of_neurons[j]):
                        self.sum[j][k] = self.scalar_sum_for_neuron(j, k)
                        self.neurons[j][k] = self.choose_activation_function(self.activation_function[j], self.sum[j][k])

            # Считаем ошибку
            value_layer = self.count_of_layers - 1
            while(value_layer >= 1):
                if(value_layer == self.count_of_layers - 1):
                    for j in range(self.count_of_neurons[len(self.count_of_neurons) - 1]):
                        if(j == self.learn_outputs[i]):
                            self.delta_neurons[value_layer][j] = 1 - self.neurons[value_layer][j]
                        else:
                            self.delta_neurons[value_layer][j] = -self.neurons[value_layer][j]
                        MSE = MSE + (copy.deepcopy(self.delta_neurons[value_layer][j]) ** 2)
                else:
                    for j in range(self.count_of_neurons[value_layer]):
                        self.delta_neurons[value_layer][j] = self.scalar(self.delta_neurons[value_layer + 1], self.synaptic_weights[value_layer][j])
                value_layer = value_layer - 1

            for j in range(self.count_of_neurons[len(self.count_of_neurons) - 1]):
                value = round(self.neurons[self.count_of_layers - 1][j])
                if((value == 0) and (j != self.learn_outputs[i])):
                    TN = TN + 1
                if((value == 0) and (j == self.learn_outputs[i])):
                    FN = FN + 1
                if((value == 1) and (j == self.learn_outputs[i])):
                    TP = TP + 1
                if((value == 1) and (j != self.learn_outputs[i])):
                    FP = FP + 1

            # Изменяем веса
            for j in range(self.count_of_layers - 1):
                for k in range(len(self.synaptic_weights[j])):
                    for l in range(len(self.synaptic_weights[j][k])):
                        self.synaptic_weights[j][k][l] = self.synaptic_weights[j][k][l] + self.speed * self.delta_neurons[j + 1][l] * self.choose_derivative_activation_function(self.activation_function[j + 1], self.sum[j + 1][l]) * self.neurons[j][k]

        MSE = MSE / len(self.learn_inputs)
        self.MSE_mas_learning.append(MSE)
        print("MSE =", MSE)
        print("TP = ", TP)
        print("FP = ", FP)
        print("FN = ", FN)
        print("TN = ", TN)
        # Точность работы алгоритма
        self.accuracy_mas_learning.append((TP + TN) / (TP + TN + FP + FN))
        print("accurancy = ", (TP + TN) / (TP + TN + FP + FN))
        try:
            print("precision = ", TP / (TP + FP))
            print("recall = ", TP / (TP + FN))
        except ZeroDivisionError:
            print("Ошибочка вышла. На этой эпохе метрики precision и recall не будут считаться :с")

    # Функция проверки результатов на заданной выборке
    def test_function(self):
        # MSE - Mean Squared Error (среднеквадратичная ошибка)
        MSE = 0

        TP = 0 # True Positive (Правильно определена 1)
        FP = 0 # False Positive (Неправильно определена 1)
        FN = 0 # False Negative (Неправильно определён 0)
        TN = 0 # True Negative (Правильно определён 0)

        for i in range(len(self.test_inputs)):
            # Высчитываем значения нейронов
            for j in range(self.count_of_layers):
                if(j == 0):
                    for k in range(self.count_of_signs):
                        self.neurons[0][k] = self.test_inputs[i][k]
                else:
                    for k in range(self.count_of_neurons[j]):
                        self.sum[j][k] = self.scalar_sum_for_neuron(j, k)
                        self.neurons[j][k] = self.choose_activation_function(self.activation_function[j], self.sum[j][k])

            # Считаем ошибку
            value_layer = self.count_of_layers - 1
            while(value_layer >= 1):
                if(value_layer == self.count_of_layers - 1):
                    for j in range(self.count_of_neurons[len(self.count_of_neurons) - 1]):
                        if(j == self.test_outputs[i]):
                            self.delta_neurons[value_layer][j] = 1 - self.neurons[value_layer][j]
                        else:
                            self.delta_neurons[value_layer][j] = -self.neurons[value_layer][j]
                        MSE = MSE + (self.delta_neurons[value_layer][j] ** 2)
                else:
                    for j in range(self.count_of_neurons[value_layer]):
                        self.delta_neurons[value_layer][j] = self.scalar(self.delta_neurons[value_layer + 1], self.synaptic_weights[value_layer][j])
                value_layer = value_layer - 1

            for j in range(self.count_of_neurons[len(self.count_of_neurons) - 1]):
                value = round(self.neurons[self.count_of_layers - 1][j])
                if((value == 0) and (j != self.test_outputs[i])):
                    TN = TN + 1
                if((value == 0) and (j == self.test_outputs[i])):
                    FN = FN + 1
                if((value == 1) and (j == self.test_outputs[i])):
                    TP = TP + 1
                if((value == 1) and (j != self.test_outputs[i])):
                    FP = FP + 1

        MSE = MSE / len(self.test_inputs)
        self.MSE_mas_test.append(MSE)
        print("MSE =", MSE)
        print("TP = ", TP)
        print("FP = ", FP)
        print("FN = ", FN)
        print("TN = ", TN)
        # Точность работы алгоритма
        self.accuracy_mas_test.append((TP + TN) / (TP + TN + FP + FN))
        print("accurancy = ", (TP + TN) / (TP + TN + FP + FN))
        try:
            print("precision = ", TP / (TP + FP))
            print("recall = ", TP / (TP + FN))
        except ZeroDivisionError:
            print("Ошибочка вышла. На этой эпохе метрики precision и recall не будут считаться :с")

    # Функция для проверки работы нейронной сети на валидационной выборке
    def validation_function(self):
        # MSE - Mean Squared Error (среднеквадратичная ошибка)
        MSE = 0

        TP = 0 # True Positive (Правильно определена 1)
        FP = 0 # False Positive (Неправильно определена 1)
        FN = 0 # False Negative (Неправильно определён 0)
        TN = 0 # True Negative (Правильно определён 0)

        for i in range(len(self.validation_inputs)):
            # Высчитываем значения нейронов
            for j in range(self.count_of_layers):
                if(j == 0):
                    for k in range(self.count_of_signs):
                        self.neurons[0][k] = self.validation_inputs[i][k]
                else:
                    for k in range(self.count_of_neurons[j]):
                        self.sum[j][k] = self.scalar_sum_for_neuron_validation(j, k)
                        self.neurons[j][k] = self.choose_activation_function(self.activation_function[j], self.sum[j][k])

            # Считаем ошибку
            value_layer = self.count_of_layers - 1
            while(value_layer >= 1):
                if(value_layer == self.count_of_layers - 1):
                    for j in range(self.count_of_neurons[len(self.count_of_neurons) - 1]):
                        if(j == self.validation_outputs[i]):
                            self.delta_neurons[value_layer][j] = 1 - self.neurons[value_layer][j]
                        else:
                            self.delta_neurons[value_layer][j] = -self.neurons[value_layer][j]
                        MSE = MSE + (self.delta_neurons[value_layer][j] ** 2)
                else:
                    for j in range(self.count_of_neurons[value_layer]):
                        self.delta_neurons[value_layer][j] = self.scalar(self.delta_neurons[value_layer + 1], self.best_synaptic_weights[value_layer][j])
                value_layer = value_layer - 1

            for j in range(self.count_of_neurons[len(self.count_of_neurons) - 1]):
                value = round(self.neurons[self.count_of_layers - 1][j])
                if((value == 0) and (j != self.validation_outputs[i])):
                    TN = TN + 1
                if((value == 0) and (j == self.validation_outputs[i])):
                    FN = FN + 1
                if((value == 1) and (j == self.validation_outputs[i])):
                    TP = TP + 1
                if((value == 1) and (j != self.validation_outputs[i])):
                    FP = FP + 1

        MSE = MSE / len(self.validation_inputs)
        self.MSE_validation = MSE
        print("MSE =", MSE)
        print("TP = ", TP)
        print("FP = ", FP)
        print("FN = ", FN)
        print("TN = ", TN)
        # Точность работы алгоритма
        self.accuracy_validation = (TP + TN) / (TP + TN + FP + FN)
        print("accurancy = ", (TP + TN) / (TP + TN + FP + FN))
        try:
            print("precision = ", TP / (TP + FP))
            print("recall = ", TP / (TP + FN))
        except ZeroDivisionError:
            print("Ошибочка вышла. На валидационной выборке метрики precision и recall не будут считаться :с")

    # Проверка датасета на пригодность к использованию
    def check_dataset(self):
        for i in range(len(self.file_matrix)):
            #print("len(self.file_matrix[i]) = ", len(self.file_matrix[i]))
            if(len(self.file_matrix[0]) != len(self.file_matrix[i])):
                print("Датасет испорчен. Количество столбцов различное")
                exit(0)

        print("Количество строк в исходном датасете: ", len(self.file_matrix))
        print("Количество столбцов в исходном датасете: ", len(self.file_matrix[0]))
        print()

    # Номализация входных и выходных данных
    def normalization(self, type_input_normalization, type_output_normalization):
        if(type_input_normalization == "linear"):
            self.learn_inputs = self.linear_matrix_normalization(self.learn_inputs)
            self.test_inputs = self.linear_matrix_normalization(self.test_inputs)
            self.validation_inputs = self.linear_matrix_normalization(self.validation_inputs)
        elif(type_input_normalization == "not_linear"):
            self.learn_inputs = self.not_linear_matrix_normalization(self.learn_inputs)
            self.test_inputs = self.not_linear_matrix_normalization(self.test_inputs)
            self.validation_inputs = self.not_linear_matrix_normalization(self.validation_inputs)
        if(type_output_normalization == "linear"):
            self.learn_outputs = self.linear_mas_normalization(self.learn_outputs)
            self.test_outputs = self.linear_mas_normalization(self.test_outputs)
            self.validation_outputs = self.linear_mas_normalization(self.validation_outputs)
        elif(type_output_normalization == "not_linear"):
            self.learn_outputs = self.not_linear_mas_normalization(self.learn_outputs)
            self.test_outputs = self.not_linear_mas_normalization(self.test_outputs)
            self.validation_outputs = self.not_linear_mas_normalization(self.validation_outputs)

    # Линейная нормализация массива
    def linear_mas_normalization(self, file_mas):
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

    # Линейная нормализация матрицы
    def linear_matrix_normalization(self, file_matrix):
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

    # Нелинейная нормализация массива
    def not_linear_mas_normalization(self, file_mas):
        a = 0.5
        average = 0
        for i in range(len(file_mas)):
            average = average + file_mas[i]
        average = average / len(file_mas)
        for i in range(len(file_matrix)):
            file_matrix[i] = 1 / (np.exp((-1) * a * (file_mas[i] - average)) + 1)
        return file_mas

    # Нелинейная нормализация матрицы
    def not_linear_matrix_normalization(self, file_matrix):
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
    def scalar(self, mas_1, mas_2):
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

    # Скалярное произведение для нейрона (при обучающей и тестовой выборках)
    def scalar_sum_for_neuron(self, value_layer, position_weight):
        sum = 0
        for i in range(self.count_of_neurons[value_layer - 1]):
            sum = sum + self.synaptic_weights[value_layer - 1][i][position_weight] * self.neurons[value_layer - 1][i]
        return sum

    # Скалярное произведение для нейрона (при валидационной выборке)
    def scalar_sum_for_neuron_validation(self, value_layer, position_weight):
        sum = 0
        for i in range(self.count_of_neurons[value_layer - 1]):
            sum = sum + self.best_synaptic_weights[value_layer - 1][i][position_weight] * self.neurons[value_layer - 1][i]
        return sum

    # Функция для выбора заданной функции активации
    def choose_activation_function(self, value_activation_function, sum):
        if(value_activation_function == "sigmoid"):
            return self.sigmoid_activation_function(sum)
        if(value_activation_function == "softsign"):
            return self.softsign_activation_function(sum)
        if(value_activation_function == "relu"):
            return self.relu_activation_function(sum)

    # Функция для выбора заданной производной от функции активации
    def choose_derivative_activation_function(self, value_activation_function, sum):
        if(value_activation_function == "sigmoid"):
            return self.derivative_sigmoid_activation_function(sum)
        if(value_activation_function == "softsign"):
            return self.derivative_softsign_activation_function(sum)
        if(value_activation_function == "relu"):
            return self.derivative_relu_activation_function(sum)

    # Функция активации (softsign)
    def softsign_activation_function(self, x):
        return (x / (1 + np.abs(x)))

    # Производная функции активации (softsign)
    def derivative_softsign_activation_function(self, x):
        if(x >= 0):
            return (1 / ((1 + x) * (1 + x)))
        else:
            return (1 / ((1 - x) * (1 - x)))

    # Функция активации (sigmoid)
    def sigmoid_activation_function(self, x):
        return (1 / (1 + np.exp(-x)))

    # Производная функция активации (sigmoid)
    def derivative_sigmoid_activation_function(self, x):
        return (np.exp(-x) / ((1 + np.exp(-x)) * (1 + np.exp(-x))))

    # Функция активации (relu)
    def relu_activation_function(self, x):
        return max(x, 0)

    # Производная функции активации (relu)
    def derivative_relu_activation_function(self, x):
        if(x >= 0):
            return 1
        else:
            return 0