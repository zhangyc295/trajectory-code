from keras.src.saving import load_model
from tensorflow.keras.optimizers import SGD, Adadelta, RMSprop, Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Masking, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Flatten
from keras import backend as K
import numpy as np
import csv
import os
import math
from math import factorial

# 现在你可以使用加载的模型进行预测或继续训练
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
import matplotlib

"""
def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n-k))


def get_bezier_curve(points):
    n = len(points) - 1
    return lambda t: sum(comb(n, i)*t**i * (1-t)**(n-i)*points[i] for i in range(n+1))


def evaluate_bezier(points, total):
    bezier = get_bezier_curve(points)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, total)])
    return new_points[:, 0], new_points[:, 1]
"""


def cubic_hermite_interpolation(valid_frame_labe,valid_x_center,invalid_frame_labe):
    f = interpolate.interp1d(valid_frame_labe, valid_x_center, kind=3)
    return f(invalid_frame_labe)


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
n_feature = 14
n_input = 2
n_feature_all = 14*2
n_feature_y = 14
Generate_length = 25
Generate_level = 3


FarTerm_model = load_model('trajectory_generation_Farterm（20221207）.h5')
FarTerm_model.summary()

LongTerm_model = load_model('trajectory_generation_longterm（20221101）.h5')
LongTerm_model.summary()


ShortTerm_model = load_model('trajectory_generation_shortterm（20221101）.h5')
ShortTerm_model.summary()

n = 1
dir = r"E:\Prediction_NV\New_try\trajectory_generation_deeplearingModel\data_test"
filenames = os.listdir(dir)

for filename in filenames:
    x_input = np.loadtxt(open(r"E:\Prediction_NV\New_try\trajectory_generation_deeplearingModel\data_test\{0}".format(filename), "rb"),
                         delimiter=",", skiprows=0)
    test_x = x_input.reshape(1, n_feature_all)

    y_pred = []
    PRED_SIZE = Generate_length
    PRED_BATCH = 1
    y_batch_r = np.zeros((PRED_SIZE, n_feature))
    y_pred_final = np.zeros(((PRED_SIZE+n_input), n_feature))



    based_x = test_x.reshape(-1, n_feature)
    for level in range(0, Generate_level):
        output_j = based_x[0, :]
        for j in range(0, based_x.shape[0]-1, PRED_BATCH):
            input_point = based_x[j:j+2, :].reshape(1, (n_input*n_feature))
            input_point1 = input_point

            print(level)
            if level == 0:
                print('长远期来了')
                y_batch = FarTerm_model.predict(input_point1)
            elif level == 1:
                print('远期来了')
                y_batch = LongTerm_model.predict(input_point1)
            else:
                print('没有来')
                y_batch = ShortTerm_model.predict(input_point1)
            y_batch = y_batch[0, :]


            y_batch[0] = input_point[0, 0] + y_batch[0] * (input_point[0, n_feature]-input_point[0, 0])
            y_batch[1] = input_point[0, 1] + y_batch[1] * (input_point[0, n_feature + 1] - input_point[0, 1])

            end_point = based_x[j+1,:]
            output_j = np.row_stack((output_j, [y_batch, end_point]))
        based_x = output_j

    all_frame = range(0, Generate_length)
    if Generate_length == 17:
       valid_frame_labe = [0, 4, 8, 12, 16]
    else:
       valid_frame_labe = [0, 3, 6, 9, 12, 15, 18, 21, 24]
    Key_Ponts = output_j[:, 0:2]






    invalid_frame_labe = list(set(all_frame) - set(valid_frame_labe))
    f_x = interpolate.interp1d(valid_frame_labe, Key_Ponts[:, 0], kind=3)
    f_y = interpolate.interp1d(valid_frame_labe, Key_Ponts[:, 1], kind=3)
    Interpolated_x = (f_x(invalid_frame_labe))
    Interpolated_y = (f_y(invalid_frame_labe))
    X_all = []
    Y_all = []
    for frame in range(0, len(valid_frame_labe)-1):
        X_all.append(Key_Ponts[frame, 0])
        X_all.append(Interpolated_x[frame * 2 + 0])
        X_all.append(Interpolated_x[frame * 2 + 1])
        Y_all.append(Key_Ponts[frame, 1])
        Y_all.append(Interpolated_y[frame * 2 + 0])
        Y_all.append(Interpolated_y[frame * 2 + 1])
    X_all.append(Key_Ponts[len(valid_frame_labe)-1, 0])
    Y_all.append(Key_Ponts[len(valid_frame_labe)-1, 1])

    bx = np.array(X_all)
    by = np.array(Y_all)


    output_j = [bx, by]



















    csvfile = open(r'E:\Prediction_NV\New_try\trajectory_generation_deeplearingModel\result_test\{0}'.format(filename), 'w', newline="")
    writer = csv.writer(csvfile)
    writer.writerows(output_j)

    csvfile.close()
    n=n+1


print('The trajectory generation is completed !')
