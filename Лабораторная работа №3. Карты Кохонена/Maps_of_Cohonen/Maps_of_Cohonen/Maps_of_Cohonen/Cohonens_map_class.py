from sklearn.manifold import TSNE # Библиотека необходима для уменьшения размерности данных
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import math

class neuron:
    x = 0 # х-координата нейрона в сетке
    y = 0 # y-координата нейрона в сетке

    synaptic_weights = [] # Синаптические веса нейрона

    clasterization_mark = 0 # Метка принадлежности нейрона к классу

    # Конструктор класса
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __init__(self, x, y, length):
        self.x = x
        self.y = y
        
        self.synaptic_weights = []
        random.seed(1)
        for i in range(length):
            self.synaptic_weights.append(2 * np.random.random() - 1)

    # Задание случайных весов
    def get_random_synaptic_weights(self, length):
        #print("length = ", length)
        random.seed(1)
        for i in range(length):
            self.synaptic_weights.append(2 * np.random.random() - 1)

    # Вычисление растояния между весами на этапе поиска победившего нейрона
    # Евклидова метрика
    def get_Euclidean_metric(self, mas):
        sum = 0
        for i in range(len(self.synaptic_weights)):
            sum = sum + (self.synaptic_weights[i] - mas[i]) ** 2
            #print(sum)
        sum = math.sqrt(sum)
        #print(type(sum))
        return sum

    # Метрика Чебышева
    def get_Chebyshev_metric(self, mas):
        max = 0
        for i in range(len(self.synaptic_weights)):
            if(math.fabs(self.synaptic_weights[i] - mas[i]) > max):
                max = math.fabs(self.synaptic_weights[i] - mas[i])
        return max

    # Вычисление квадрата растояния между нейронами
    def get_distance(self, x, y):
        return ((self.x - x) ** 2 + (self.y - y) ** 2) ** 2

class Cohonen_map:
    count_of_neurons_x_y = [] # Количество нейронов в сетке (x, y)
    count_of_neurons = 0 # Общее количество нейронов
    neurons = [] # Список со значениями нейронов

    metric_type = "" # Тип метрики для подсчёта расстояния между нейронами

    file_matrix = [] # Переменная для хранения исходного датасета

    classes = [] # Классы в исходном датасете
    count_of_classes = 0 # Количество классов в исходном датасете
    count_of_signs = 0 # Количество признаков в исходном датасете

    count_of_eras = 0 # Количество эпох обучения

    count_of_clasters = 0 # Количество классов (кластеров в карте Кохонена)

    # Параметры карты Кохонена
    sigma_n = 0
    h = [] # Матрица (список списков) топологических окрестностей

    sigma_0 = 1.1 # Параметр системы. Начальное значение примерно равно радиусу решётки
    tau_1 = 1000 / math.log(sigma_0, 10) # Параметр системы. Основание не точное
    tau_2 = 1000 # Параметр системы. Не должен опускаться ниже 0.01

    speed_0 = 0.1 # Начальное значение скорости обучения персептрона (параметр системы)
    speed_value = 0 # Значение скорости обучения персептрона в текущей выборке

    R_const = 5 # Параметр системы. Значение радиуса шара, необходимом для кластеризации элементов

    # Конструктор класса
    def __init__(self, dataset_path, count_of_neurons, count_of_eras, speed, metric_type, R):
        self.read_from_file(dataset_path)

        self.count_of_neurons_x_y = count_of_neurons
        self.count_of_neurons = self.count_of_neurons_x_y[0] * self.count_of_neurons_x_y[1]
        self.count_of_eras = count_of_eras
        self.speed = speed
        self.metric_type = metric_type
        self.R_const = R

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

    # Функция для отрисовки нейронов
    def draw_neurons(self):
        x = []
        y = []
        weights = []
        counter = 0
        for i in range(self.count_of_neurons_x_y[0]):
            for j in range(self.count_of_neurons_x_y[1]):
                x.append(self.neurons[counter].x)
                y.append(self.neurons[counter].y)
                weights.append(self.neurons[counter].synaptic_weights)
                counter = counter + 1

        model = TSNE(learning_rate = 100)
        colors_3d = model.fit_transform(weights)
        print("colors_3d = ")
        print(colors_3d)
        print()

        colors_3d = self.linear_matrix_normalization(colors_3d)

        fig = plt.figure()

        figure_points = fig.add_subplot(111)
        figure_points.set_title("Точки для двумерных исходных данных")

        for i in range(self.count_of_neurons):
            figure_points.scatter(self.neurons[i].x, self.neurons[i].y, color = [colors_3d[i][0], colors_3d[i][1], 0])

        plt.show()

    # Функция для отрисовки нейронов после кластеризации
    def draw_neurons_after_clasterization(self):
        x = []
        y = []
        weights = []
        color_claster = []
        counter = 0
        for i in range(self.count_of_neurons_x_y[0]):
            for j in range(self.count_of_neurons_x_y[1]):
                x.append(self.neurons[counter].x)
                y.append(self.neurons[counter].y)
                weights.append(self.neurons[counter].synaptic_weights)
                color_claster.append(self.neurons[counter].clasterization_mark)
                counter = counter + 1

        # Уменьшение размерности данных для весов
        model = TSNE(learning_rate = 100)
        colors_3d = model.fit_transform(weights)
        print("colors_3d = ")
        print(colors_3d)
        print()

        colors_3d = self.linear_matrix_normalization(colors_3d)

        color_claster = self.linear_mas_normalization(color_claster)

        fig = plt.figure()

        figure_points = fig.add_subplot(111)
        figure_points.set_title("Нейроны после кластеризации")

        for i in range(self.count_of_neurons):
            figure_points.scatter(self.neurons[i].x, self.neurons[i].y, color = [color_claster[i], color_claster[i], color_claster[i]])

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

    # Удаление значений классов из исходного датасета
    def delete_class_from_dataset(self):
        for i in range(len(self.file_matrix)):
            self.file_matrix[i].pop()
        self.count_of_signs = len(self.file_matrix[0])

    # Инициализация нейронов и h и задание случайных весов
    def neuron_initialization(self):
        # Инициализация матрицы (списка) нейронов
        for i in range(self.count_of_neurons_x_y[0]):
            value_mas = []
            for j in range(self.count_of_neurons_x_y[1]):
                self.neurons.append(neuron(i, j, self.count_of_signs))
                value_mas.append(0)
            self.h.append(copy.deepcopy(value_mas))

    # Функция обучения карты Кохонена
    def learning_procedure(self):
        era_counter = 0 # Счётчик эпох

        # Процесс обучения
        while(era_counter <= self.count_of_eras):
            #if(era_counter % 2500 == 0):
            #    self.draw_neurons()

            print("Эпоха обучения №", era_counter + 1)
            print()

            number_value_input = math.floor(np.random.random() * len(self.file_matrix))
            
            # Поиск победившего нейрона
            x_neuron_min = 0
            y_neuron_min = 0
            min_sum = self.neurons[0].get_Euclidean_metric(self.file_matrix[number_value_input])
            counter = 0

            if(self.metric_type == "euclidean"):
                for i in range(self.count_of_neurons_x_y[0]):
                    for j in range(self.count_of_neurons_x_y[1]):
                        if(self.neurons[counter].get_Euclidean_metric(self.file_matrix[number_value_input]) < min_sum):
                            min_sum = self.neurons[counter].get_Euclidean_metric(self.file_matrix[number_value_input])
                            x_neuron_min = self.neurons[counter].x
                            y_neuron_min = self.neurons[counter].y
                        counter = counter + 1
            elif(self.metric_type == "chebyshev"):
                for i in range(self.count_of_neurons_x_y[0]):
                    for j in range(self.count_of_neurons_x_y[1]):
                        if(self.neurons[counter].get_Chebyshev_metric(self.file_matrix[number_value_input]) < min_sum):
                            min_sum = self.neurons[counter].get_Euclidean_metric(self.file_matrix[number_value_input])
                            x_neuron_min = self.neurons[counter].x
                            y_neuron_min = self.neurons[counter].y
                        counter = counter + 1
            
            # Вычисление значений d_j_i и h_j_i
            self.sigma_n = self.sigma_0 * math.exp(- counter / self.tau_1)

            counter = 0
            for i in range(self.count_of_neurons_x_y[0]):
                for j in range(self.count_of_neurons_x_y[1]):
                    self.h[i][j] = math.exp(-(self.neurons[counter].get_distance(self.neurons[x_neuron_min * self.count_of_neurons_x_y[0] + y_neuron_min].x, self.neurons[x_neuron_min * self.count_of_neurons_x_y[0] + y_neuron_min].y)) / (2 * self.sigma_n))
                    counter = counter + 1

            # Изменение синаптических весов (процесс адаптации)
            self.speed_value = self.speed * math.exp(- counter / self.tau_2)
    
            counter = 0
            for i in range(self.count_of_neurons_x_y[0]):
                for j in range(self.count_of_neurons_x_y[1]):
                    for k in range(self.count_of_signs):
                        self.neurons[counter].synaptic_weights[k] = self.neurons[counter].synaptic_weights[k] + self.speed_value * self.h[i][j] * (self.file_matrix[number_value_input][k] - self.neurons[counter].synaptic_weights[k])
                    counter = counter + 1

            era_counter = era_counter + 1

        self.draw_neurons()

    # Функция кластеризации карты Кохонена
    def ForEL_clasterization_function(self):
        era_counter = 0 # Счётчик эпох
        clasterization_mark = 0 # Метка при кластеризации

        value_neurons = copy.deepcopy(self.neurons)
        count_value_neurons = len(value_neurons)

        number_neuron_center = math.floor(np.random.random() * count_value_neurons)

        # Процесс обучения
        while(count_value_neurons > 0):

            print("Эпоха кластеризации №", era_counter + 1)
            print()

            # Помечаем объекты выборки (если в шаре - номер кластера, иначе -1)
            for i in range(count_value_neurons):
                if(value_neurons[i].get_Euclidean_metric(value_neurons[number_neuron_center].synaptic_weights) < self.R_const):
                    value_neurons[i].clasterization_mark = clasterization_mark
                else:
                    value_neurons[i].clasterization_mark = -1

            # Находим номер нейрона с новым центром шара
            sum_synaptic_weights = []
            
            for i in range(self.count_of_signs):
                sum_synaptic_weights.append(0)

            # Находим среднее значение синаптических весов
            counter = 0
            for i in range(count_value_neurons):
                if(value_neurons[i].clasterization_mark == clasterization_mark):
                    sum_synaptic_weights = sum_synaptic_weights + value_neurons[i].synaptic_weights
                    counter = counter + 1

            for i in range(len(sum_synaptic_weights)):
                sum_synaptic_weights[i] = sum_synaptic_weights[i] / counter

            min = -1 # Минимальное значение текущего центра
            i_min = -1 # Номер текущего центра
            for i in range(count_value_neurons):
                if(value_neurons[i].clasterization_mark == clasterization_mark):
                    if((value_neurons[number_neuron_center].get_Euclidean_metric(value_neurons[i].synaptic_weights) < min) or (i_min == - 1)):
                        min = value_neurons[number_neuron_center].get_Euclidean_metric(value_neurons[i].synaptic_weights)
                        i_min = i

            if(i_min == number_neuron_center):
                i = 0
                while(i < count_value_neurons):
                    if(value_neurons[i].clasterization_mark == clasterization_mark):
                        for j in range(self.count_of_neurons):
                            if((value_neurons[i].x == self.neurons[j].x) and (value_neurons[i].y == self.neurons[j].y)):
                                self.neurons[j].clasterization_mark = clasterization_mark
                        value_neurons.pop(i)
                        count_value_neurons = count_value_neurons - 1
                        i = i - 1
                    i = i + 1
                clasterization_mark = clasterization_mark + 1
                number_neuron_center = math.floor(np.random.random() * count_value_neurons)
            else:
                number_neuron_center = i_min

            era_counter = era_counter + 1

        print("Кластеризация окончена")
        self.count_of_clasters = clasterization_mark
        print("Количество клайстеров в карте Кохонена - ", self.count_of_clasters)
        self.draw_neurons_after_clasterization()

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
    def normalization(self, normalization_type):
        if(normalization_type == "linear"):
            self.linear_matrix_normalization(self.file_matrix)
        elif(normalization_type == "not_linear"):
            self.not_linear_matrix_normalization()

    # Линейная нормализация массива
    def linear_mas_normalization(self, file_mas):
        max = file_mas[0]
        min = file_mas[0]
        for i in range(len(file_mas)):
            if(file_mas[i] > max):
                max = file_mas[i]
            if(file_mas[i] < min):
                min = file_mas[i]
        if(min != max):
            for i in range(len(file_mas)):
                file_mas[i] = (file_mas[i] - min) / (max - min)
        #print("Нормализованный исходный массив")
        #print(file_mas)
        return file_mas

    # Линейная нормализация матрицы
    def linear_matrix_normalization(self, matrix):
        for i in range(len(matrix[0])):
            max = matrix[0][i]
            min = matrix[0][i]
            for j in range(len(matrix)):
                if(matrix[j][i] > max):
                    max = matrix[j][i]
                if(matrix[j][i] < min):
                    min = matrix[j][i]
            for j in range(len(matrix)):
                matrix[j][i] = (matrix[j][i] - min) / (max - min)
        return matrix

    # Нелинейная нормализация матрицы
    def not_linear_matrix_normalization(self):
        a = 0.5 # Коэффициент нормализации
        for i in range(len(self.file_matrix[0])):
            average = 0
            for j in range(len(self.file_matrix)):
                average = average + self.file_matrix[j][i]
            average = average / len(self.file_matrix)
            for j in range(len(self.file_matrix)):
                self.file_matrix[j][i] = 1 / (np.exp((-1) * a * (self.file_matrix[j][i] - average)) + 1)

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
        for i in range(self.count_of_neurons_x_y[value_layer - 1]):
            sum = sum + self.synaptic_weights[value_layer - 1][i][position_weight] * self.neurons[value_layer - 1][i]
        return sum

    # Скалярное произведение для нейрона (при валидационной выборке)
    def scalar_sum_for_neuron_validation(self, value_layer, position_weight):
        sum = 0
        for i in range(self.count_of_neurons_x_y[value_layer - 1]):
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