#import numpy as np
#import random
#import matplotlib.pyplot as plt
#import copy
#import math

## Функция для отрисовки массива точек на графике
#def draw_points(points_matrix):
#    fig = plt.figure()

#    x1 = []
#    y1 = []

#    for i in range(len(points_matrix)):
#        x1.append(points_matrix[i][0])
#        y1.append(points_matrix[i][1])

#    figure_points = fig.add_subplot(111)
#    figure_points.set_title("Точки для двумерных исходных данных")
#    figure_points.scatter(x1, y1, color = [0, 0, 0])

#    plt.show()

#matrix = [[0, 0], [1, 1], [3, 2], [2, 1]]
#draw_points(matrix)

for i in range(10):
    print(i)
    if(i == 5):
        i = i + 5