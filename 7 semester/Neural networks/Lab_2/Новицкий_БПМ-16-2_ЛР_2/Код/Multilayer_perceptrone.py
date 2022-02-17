# Количество нейронов равно количеству признаков
# Все признаки в исходном датасете - целые числа в промежутке [0; n], где n - количество классов + 1
import class_perceptrone
import csv
import numpy as np
import random
import math
import copy

# Открытие файла через проводник
#import easygui
#csv_path = easygui.fileopenbox()

# Основное тело кода
# Считывание датасета из файла
my_dataset_path = "dataset/htru_2.csv"
Roma_dataset_path = "dataset/data_banknote_authentication_Roma_dataset.txt"
Alyona_dataset_path = "dataset/avila_all_Alyona_dataset.txt"
iris_dataset_path = "dataset/Iris.txt"
generated_dataset_1_path = "dataset/generated_dataset_1.txt"
new_generated_dataset_path = "dataset/new_generated_dataset.txt"

count_of_layers = 4 # Общее количество слоёв в нейронной сети
neurons_hidden = [8, 8] # Количество нейронов в скрытых слоях

# Количество % из всего датасета для тестового множества
test_probability = 20
# Количество % из всего датасета для валидационного множества
validation_probability = 10

# Функции активации
activation_function = "sigmoid"
activation_function = ["sigmoid", "sigmoid", "sigmoid", "sigmoid"] # Функции активации для каждого слоя
#activation_function = "softsign"

count_of_eras = 25 # Количество эпох
speed = 0.1 # Скорость обучения

percentrone = class_perceptrone.multilayer_perceptrone(my_dataset_path, count_of_layers, test_probability, validation_probability, activation_function, neurons_hidden, count_of_eras, speed)
percentrone.read_from_file(my_dataset_path)

# Проверка датасета на корректность
percentrone.check_dataset()

# Конвертация значений исходной матрицы в тип float
percentrone.converting()

# Если двумерные данные, то рисуем их на плоскости
if(len(percentrone.file_matrix) - 1 == 2):
    perceptrone.draw_points(file_matrix)

# Определение количества классов, количества нейронов и количества весов
# Количество нейронов в выходном слое равно количеству весов
percentrone.get_classes()

# Генерация случайных значений для весов
percentrone.get_random_synaptic_weights()

# Разделение исходного датасета на множества случайным способом
percentrone.devide_learning_test_validation()

# Нормализация данных
type_input_normalization = "not_linear"
#type_input_normalization = ""
type_output_normalization = ""

percentrone.normalization(type_input_normalization, type_output_normalization)

# Единичный вход для нейронов
percentrone.single_inputs_for_neurons()

# Обучение персептрона
percentrone.learning_procedure()