# Количество нейронов равно количеству признаков
# Все признаки в исходном датасете - целые числа в промежутке [0; n], где n - количество классов + 1
import Cohonens_map_class
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

count_of_neurons = [8, 8] # Количество нейронов в сетке Кохонена

count_of_eras = 1000 # Количество эпох
speed = 0.1 # Скорость обучения
metric_type = "euclidean"
#metric_type = "chebyshev"

R = 2 # Радиус для кластеризации

map = Cohonens_map_class.Cohonen_map(my_dataset_path, count_of_neurons, count_of_eras, speed, metric_type, R)

# Проверка датасета на корректность
map.check_dataset()

# Конвертация значений исходной матрицы в тип float
map.converting()

# Удаляем значения классов из исходного датасета и определяем количество признаков
map.delete_class_from_dataset()

# Инициализация нейронов и задание случайных синаптических весов
map.neuron_initialization()

# Если двумерные данные, то рисуем их на плоскости
if(len(map.file_matrix) - 1 == 2):
    perceptrone.draw_points(file_matrix)

# Нормализация данных
normalization_type = "not_linear"
map.normalization(normalization_type)

# Обучение карты Кохонена
map.learning_procedure()

# Кластеризация данных
map.ForEL_clasterization_function()