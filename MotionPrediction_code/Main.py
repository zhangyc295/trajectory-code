import argparse
import ast
import os
import torch
import numpy as np
import pandas as pd
import DataSetProcessing
import Prediction
import PredictionEvaluation
import matplotlib.pyplot as plt
import Trajectory_Generation_Train
import csv
import openpyxl


def get_parser():
    parser = argparse.ArgumentParser(
        description='Data-Knowledge-Integration-MotionPrediction')
    parser.add_argument('--Num_B', default=6, type=int)
    parser.add_argument('--Num_V', default=6, type=int)
    parser.add_argument('--generating_lengh', default=25, type=int)
    parser.add_argument('--data_dir',  type=str, default=
                        'E:\\Uncertainty_Analysis_of_Cycling_Behavior\\Methodology\\'
                        'Data-Knowledge-Integration-MotionPrediction')
    parser.add_argument('--Pred_length', type=int, default=25)
    parser.add_argument('--rep', type=int, default=0.4)
    parser.add_argument('--train', type=str, default=False)
    return parser


def data_loader(filepath):

    e1 = pd.read_excel(filepath, sheet_name=0)
    e1_v = pd.read_excel(filepath, sheet_name=1)
    w1_v = pd.read_excel(filepath, sheet_name=2)

    return e1, e1_v, w1_v


def data_processing(e1, e1_v, w1_v, Num_B, Num_V):

    e1 = np.delete(e1, 13, axis=1)
    e1_v = np.delete(e1_v, 13, axis=1)
    w1_v = np.delete(w1_v, 13, axis=1)
    e1 = np.hstack((e1[:, :11], e1[:, 12:13], e1[:, 11:12], e1[:, 13:]))
    e1_v = np.hstack((e1_v[:, :11], e1_v[:, 12:13], e1_v[:, 11:12], e1_v[:, 13:]))
    w1_v = np.hstack((w1_v[:, :11], w1_v[:, 12:13], w1_v[:, 11:12], w1_v[:, 13:]))


    e1 = change_rad(e1, 90)
    e1_v = change_rad(e1_v, 90)
    w1_v = change_rad(w1_v, 90)


    e1 = same_out(e1)
    e1_v = same_out(e1_v)
    w1_v = same_out(w1_v)


    E_straight = np.hstack((e1[:, :12], e1[:, 13:], e1[:, 12:13]))


    NVinterract, Vinterract = DATA_merge(e1, [], e1_v, w1_v, [])


    E_trajectory = data_handle(E_straight, NVinterract, Vinterract, Num_B, Num_V, 1)

    return E_trajectory


def change_rad(e1, VAL):


    for i in range(e1.shape[0]):
        for j in range(1, 8, 2):
            theta, r = np.arctan2(e1[i, j + 1], e1[i, j]), np.sqrt(e1[i, j] ** 2 + e1[i, j + 1] ** 2)
            x_new, y_new = np.cos(np.deg2rad(VAL)+theta)*r, np.sin(np.deg2rad(VAL)+theta)*r
            e1[i, j], e1[i, j + 1] = x_new, y_new
    return e1


def same_out(e1):

    index = int(np.max(e1[:, 12]))
    e = []
    for i in range(int(e1[-1, 12])):
        alone = np.zeros((1, index*14))
        current = e1[e1[:, 12] == i + 1, :]
        current = current[:, :-1]
        current = current.reshape(1, -1)
        alone[0, :current.shape[1]] = current
        e.append(alone)
    e = np.vstack(e)
    e_handle = np.unique(e, axis=0)
    output_e = np.array([])
    index = 1
    for i in range(e_handle.shape[0]):
        if e_handle.shape[1] % 13 != 0:
            alone = np.hstack([e_handle[i, :], np.zeros(13 - e_handle.shape[1] % 13)])
        else:
            alone = e_handle[i, :]
        current_after = alone.reshape(-1, 13)
        current_after = current_after[current_after[:, 0] != 0]
        current_after = np.hstack([current_after[:, :-1], np.ones((current_after.shape[0], 1))*index, current_after[:, -1:]])
        if current_after.size != 0:
            output_e = np.vstack([output_e, current_after])
            index += 1
    return output_e


def DATA_merge(e1, n1, e1_v, n1_v, S):

    NVinterract = e1.copy()
    if n1.size != 0:
        n1[:, 12] += int(NVinterract[-1, 12])
        NVinterract = np.vstack([NVinterract, n1])
    if S.size != 0:
        S[:, 12] += int(NVinterract[-1, 12])
        NVinterract = np.vstack([NVinterract, S])

    Vinterract = e1_v.copy()
    n1_v[:, 12] += int(Vinterract[-1, 12])
    Vinterract = np.vstack([Vinterract, n1_v])

    return NVinterract, Vinterract


def data_handle(E_straight, NVinterract, Vinterract, Num_B, Num_V, index_zeros):
    """
    E_straight is the prediction subject, and the next five matrices
    are the interaction objects, from which the interaction is selected；
    """
    print('start data_handle')


    Nu_NVinterract = np.zeros((E_straight.shape[0], 1))
    all_NVinterract = np.zeros((E_straight.shape[0], Num_B * 11))
    chosed_NVinterract = np.zeros((E_straight.shape[0], Num_B * 11))

    for i in range(E_straight.shape[0]):
        nu = 0
        for i1 in range(NVinterract.shape[0]):
            if (NVinterract[i1, 0] == E_straight[i, 0]
                    and NVinterract[i1, 1] != E_straight[i, 1]
                    and NVinterract[i1, 2] != E_straight[i, 2]):
                nu += 1
                all_NVinterract[i, ((nu * 11) - 10):(nu * 11)] = NVinterract[i1, 1:10]
                all_NVinterract[i, (nu * 11)] = NVinterract[i1, 13]
        Nu_NVinterract[i, 0] = nu
        d = np.zeros((2, max(nu, Num_B)))
        for i2 in range(1, max(nu, Num_B)):
            if all_NVinterract[i, (i2 * 11) - 10] != 0:
                d[0, i2] = i2
                d[1, i2] = (((E_straight[i, 1] - all_NVinterract[i, (i2 * 11) - 10]) ** 2 +
                              (E_straight[i, 2] - all_NVinterract[i, (i2 * 11) - 9]) ** 2) ** 0.5)
        d[:, (np.where(d[1, :] == 0)[0])] = []
        d = d[:, d[1, :].argsort()]
        for i3 in range(1, min(Num_B, d.shape[1])):
            chosed_NVinterract[i, (i3 * 11 - 10):(i3 * 11 - 1)] = all_NVinterract[i, ((d[0, i3] * 11) - 10):(d[0, i3] * 11) - 1]
            chosed_NVinterract[i, i3 * 11] = all_NVinterract[i, (d[0, i3] * 11)]



    Nu_Vinterract = np.zeros((E_straight.shape[0], 1))
    all_Vinterract = np.zeros((E_straight.shape[0], Num_V * 11))
    chosed_Vinterract = np.zeros((E_straight.shape[0], Num_V * 11))
    for i in range(E_straight.shape[0]):
        nu_V = 0
        for i1 in range(Vinterract.shape[0]):
            if (Vinterract[i1, 0] == E_straight[i, 0]
                    and Vinterract[i1, 1] != E_straight[i, 1]
                    and Vinterract[i1, 2] != E_straight[i, 2]):
                nu_V += 1
                all_Vinterract[i, ((nu_V * 11) - 10):(nu_V * 11)] = Vinterract[i1, 1:10]
                all_Vinterract[i, (nu_V * 11)] = Vinterract[i1, 13]
        Nu_Vinterract[i, 0] = nu_V
        d2 = np.zeros((2, max(nu_V, Num_V)))
        for i2 in range(1, max(nu_V, Num_V)):
            if all_Vinterract[i, (i2 * 11) - 10] != 0:
                d2[0, i2] = i2
                d2[1, i2] = (((E_straight[i, 1] - all_Vinterract[i, (i2 * 11) - 10]) ** 2 +
                               (E_straight[i, 2] - all_Vinterract[i, (i2 * 11) - 9]) ** 2) ** 0.5)
        d2[:, (np.where(d2[1, :] == 0)[0])] = []
        d2 = d2[:, d2[1, :].argsort()]
        for i3 in range(1, min(Num_V, d2.shape[1])):
            chosed_Vinterract[i, (i3 * 11 - 10):(i3 * 11 - 1)] = all_Vinterract[i, ((d2[0, i3] * 11) - 10):(d2[0, i3] * 11) - 1]
            chosed_Vinterract[i, i3 * 11] = all_Vinterract[i, (d2[0, i3] * 11)]


    E = np.zeros((E_straight.shape[0], 14 + Num_B * 11 + Num_V * 11))
    E[:, 0] = E_straight[:, 13]
    E[:, 1] = E_straight[:, 11]
    E[:, 2:5] = E_straight[:, :3]
    E[:, 5:13] = E_straight[:, 3:11]
    E[:, 13] = E_straight[:, 12]
    E[:, 14:(14 + Num_B * 11)] = chosed_NVinterract[:, :(Num_B * 11)]
    E[:, (14 + Num_B * 11):(14 + Num_B * 11 + Num_V * 11)] = chosed_Vinterract[:, :(Num_V * 11)]


    output_E = E
    return output_E


def Data_saving(model_path,  min_boundary, max_boundary, x_train, y_train, x_ver, y_ver, x_test, y_test, test_id):

    train_data_path = os.path.join(model_path, "train_data")
    test_data_path = os.path.join(model_path, "test_data")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)
    np.savetxt(os.path.join(train_data_path, "x_train.csv"), x_train, delimiter=",")
    np.savetxt(os.path.join(train_data_path, "y_train.csv"), y_train, delimiter=",")
    np.savetxt(os.path.join(train_data_path, "x_ver.csv"), x_ver, delimiter=",")
    np.savetxt(os.path.join(train_data_path, "y_ver.csv"), y_ver, delimiter=",")
    np.savetxt(os.path.join(train_data_path, "x_test.csv"), x_test, delimiter=",")
    np.savetxt(os.path.join(train_data_path, "y_test.csv"), y_test, delimiter=",")
    np.savetxt(os.path.join(model_path, "min_boundary.csv"), min_boundary, delimiter=",")
    np.savetxt(os.path.join(model_path, "max_boundary.csv"), max_boundary, delimiter=",")
    np.savetxt(os.path.join(model_path, "test_id.csv"), test_id, delimiter=",")

    for i in range(x_test.shape[0]):
        file_path = os.path.join(test_data_path, 'x_test_' + str(i + 1) + '.txt')
        np.savetxt(file_path, x_test[i, :], delimiter=',')


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    """
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.set_device(0)
    """

    dir_path = 'data'
    filename = 'E_trajectory.csv'

    file_path = os.path.join(dir_path, filename)

    if os.path.isfile(file_path):

        df = pd.read_csv(file_path, delimiter=',')

        E_trajectory = df.to_numpy()
        print('loading。。。')
        print(E_trajectory.shape)

    else:

        e1, e1_v, w1_v = data_loader(r'E:\zycpostgraduate\非机动车轨迹预测\data_XXRoad.xlsx')

        E_trajectory = data_processing(e1, e1_v, w1_v, args.Num_B, args.Num_V)
        print("None")

    E_trajectory_1 = E_trajectory[(E_trajectory[:, 1] >= 1) & (E_trajectory[:, 1] <= 10), :25]

    print(args.train)
    if args.train is True:


        dp1 = DataSetProcessing.DataSetProcessing(E_trajectory_1, 4, 7, args)
        min_boundary, max_boundary, x_train_1, y_train_1, x_ver_1, y_ver_1, x_test_1, y_test_1, test_id = dp1.data_dividing()
        dp2 = DataSetProcessing.DataSetProcessing(E_trajectory_1, 8, 15, args)
        _, _, x_train_2, y_train_2, x_ver_2, y_ver_2, x_test_2, y_test_2, _ = dp2.data_dividing()


        data_dir1 = os.path.join(args.data_dir, "Traj_Gen_model_1")
        data_dir2 = os.path.join(args.data_dir, "Traj_Gen_model_2")
        Data_saving(data_dir1, min_boundary, max_boundary, x_train_1,
                    y_train_1, x_ver_1, y_ver_1, x_test_1, y_test_1, test_id)
        Data_saving(data_dir2, min_boundary, max_boundary, x_train_2,
                    y_train_2, x_ver_2, y_ver_2, x_test_2, y_test_2, test_id)


        if os.path.exists(os.path.join(data_dir1, "trajectory_generation_model.h5")):
            print("Model 1 exists")
        else:
            Trajectory_Generation_Train.Trajectory_Generation_Train(data_dir1)
        if os.path.exists(os.path.join(data_dir2, "trajectory_generation_model.h5")):
            print("Model 2 exists")
        else:
            print("文件不存在，开始训练模型2")
            Trajectory_Generation_Train.Trajectory_Generation_Train(data_dir2)


        if args.Pred_length * 0.12 > 2:
            data_dir3 = os.path.join(args.data_dir, "Traj_Gen_model_3")
            dp3 = DataSetProcessing.DataSetProcessing(E_trajectory_1, 16, 27, args)
            _, _, x_train_3, y_train_3, x_ver_3, y_ver_3, x_test_3, y_test_3, _ = dp3.data_dividing()
            Data_saving(data_dir3, min_boundary, max_boundary, x_train_3,
                        y_train_3, x_ver_3, y_ver_3, x_test_3, y_test_3, test_id)
            if os.path.exists(os.path.join(data_dir3, "trajectory_generation_model.h5")):
                print("模型3已存在")
            else:
                print("文件不存在，开始训练模型3")
                Trajectory_Generation_Train.Trajectory_Generation_Train(data_dir3)




    path_sample = 'E:/Uncertainty_Analysis_of_Cycling_Behavior/Methodology' \
                  '/Data-Knowledge-Integration-MotionPrediction/Sample'
    test_id = np.loadtxt(open(os.path.join(path_sample, "test_id.txt"), "rb"), delimiter="\t", skiprows=0)
    max_boundary = np.loadtxt(open(os.path.join(path_sample, "mmax(sample).txt"), "rb"), delimiter="\t", skiprows=0)
    min_boundary = np.loadtxt(open(os.path.join(path_sample, "mmin(sample).txt"), "rb"), delimiter="\t", skiprows=0)
    max_boundary = np.reshape(max_boundary, (1, -1))
    min_boundary = np.reshape(min_boundary, (1, -1))

    pre = Prediction.TwoLayerModelTest(0.8, test_id, args, max_boundary, min_boundary, E_trajectory)
    test_result_no, all_exit_flag, where_opt = pre.test()
    plt.hist(np.array(where_opt))
    plt.show()


    ade, fde = PredictionEvaluation.error_sample(args, E_trajectory, test_result_no)
    print("ADE ：", ade)
    print("FDE ：", fde)


