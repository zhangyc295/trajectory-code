import numpy as np
import DataExpand
import dill

class DataSetProcessing:
    def __init__(self, input1, low, high, args):
        self.input1 = input1
        self.low = low
        self.high = high
        self.args = args

    def data_dividing(self):
        p_test = 0.1
        p_train = 0.7

        all_ID = np.unique(self.input1[:, 0])
        t = round(p_test * all_ID.shape[0])
        nu = np.random.permutation(all_ID.shape[0])
        all_ID = all_ID[nu]
        data = self.input1.copy()
        data_test = np.empty([0, data.shape[1]])
        data_train_ver = np.empty([0, data.shape[1]])
        for i in range(all_ID.shape[0]):
            print(i)
            if i <= t:
                data_test = np.concatenate((data_test, data[np.where(data[:, 0] == all_ID[i])]), axis=0)
            else:
                data_train_ver = np.concatenate((data_train_ver, data[np.where(data[:, 0] == all_ID[i])]), axis=0)

        sample_test = self.data_expand_test(data_test)
        sample_train_ver = self.data_expand(data_train_ver, self.low, self.high)
        test_id = sample_test[::25, :3]
        N_lengh_point = sample_test.shape[1]
        N_lengh_train = 3 * N_lengh_point
        N_lengh_test = N_lengh_point * self.args.Pred_length

        precessing_test = np.reshape(sample_test, (-1, N_lengh_test))
        precessing_train_ver = np.reshape(sample_train_ver, (-1, N_lengh_train))
        rrnu = np.random.permutation(precessing_train_ver.shape[0])
        precessing_train_ver = precessing_train_ver[rrnu, :]

        precessing_test = np.reshape(precessing_test, (-1, N_lengh_point))
        precessing_train_ver = np.reshape(precessing_train_ver, (-1, N_lengh_point))

        precessing_train_ver_output, precessing_test_output = self.replace_feature_od_points(precessing_train_ver,
                                                                               precessing_test,
                                                                               self.args.Pred_length)

        char_all = np.concatenate((precessing_train_ver_output, precessing_test_output), axis=0)
        char_all = np.delete(char_all, np.s_[:4], axis=1)

        char_all_handle = char_all.copy()
        char_all_handle = np.delete(char_all_handle, np.s_[6:10], axis=1)
        char_all_handle = np.delete(char_all_handle, np.s_[13:17], axis=1)

        output, mmin, mmax = self.normalizing(char_all_handle)

        lin_train_ver = precessing_train_ver.shape[0]
        N_lengh_point = output.shape[1]
        N_lengh_train = 3 * N_lengh_point
        N_lengh_test = N_lengh_point * self.args.Pred_length

        data_train_ver_Nor = output[0:lin_train_ver, :]
        data_train_ver_Nor = np.reshape(data_train_ver_Nor.T, (N_lengh_train, -1)).T

        data_test_Nor = output[lin_train_ver:output.shape[0], :]
        data_test_Nor = np.reshape(data_test_Nor.T, (N_lengh_test, -1)).T

        tr = round(p_train * data_train_ver_Nor.shape[0])

        x_test = np.concatenate((data_test_Nor[:, 0:N_lengh_point],
                          data_test_Nor[:, N_lengh_test - N_lengh_point:N_lengh_test]), axis=1)
        y_test = data_test_Nor[:, N_lengh_point:N_lengh_test - N_lengh_point]

        x_train = np.concatenate((data_train_ver_Nor[0:tr, 0:N_lengh_point],
                           data_train_ver_Nor[0:tr, 2 * N_lengh_point:N_lengh_train]), axis=1)
        y_train = data_train_ver_Nor[0:tr, N_lengh_point:2 * N_lengh_point]

        x_ver = np.concatenate((data_train_ver_Nor[tr:data_train_ver_Nor.shape[0], 0:N_lengh_point],
                         data_train_ver_Nor[tr:data_train_ver_Nor.shape[0], 2 * N_lengh_point:N_lengh_train]),
                        axis=1)
        y_ver = data_train_ver_Nor[tr:data_train_ver_Nor.shape[0], N_lengh_point:2 * N_lengh_point]

        y_train, x_train = self.handle_y(y_train, x_train)
        y_ver, x_ver = self.handle_y(y_ver, x_ver)

        return mmin, mmax, x_train, y_train, x_ver, y_ver, x_test, y_test, test_id

    @staticmethod
    def replace_feature_od_points(precessing_train_ver, precessing_test, generating_lengh):
        precessing_train_ver1 = precessing_train_ver.copy()
        precessing_test1 = precessing_test.copy()
        n = 11
        ID_Data_train_ver = precessing_train_ver1[:, 0:4]
        ID_Data_test = precessing_test1[:, 0:4]
        precessing_train_ver1 = precessing_train_ver1[:, 4:]
        precessing_test1 = precessing_test1[:, 4:]
        precessing_train_ver1 = np.reshape(precessing_train_ver1.T, (3 * n * 2, -1),
                                           order='F').T
        precessing_test1 = np.reshape(precessing_test1.T, (generating_lengh * n * 2, -1), order='F').T
        timestep = 1.388888888884110e-06
        T_start = precessing_train_ver[(np.arange(0, precessing_train_ver.shape[0]-2, 3)), 3]
        T_end = precessing_train_ver[(np.arange(0, precessing_train_ver.shape[0]-2, 3)) + 1, 3]
        T = 0.12 * ((T_end - T_start) / timestep)
        T = T.reshape(-1, 1)
        T = np.hstack((T, T))
        Host_S_start = precessing_train_ver1[:, 0:2]
        Host_V_start = precessing_train_ver1[:, 2:4]
        Host_A_start = precessing_train_ver1[:, 4:6]
        Host_S_end = precessing_train_ver1[:, 44:46]
        Host_V_end = (2. * (Host_S_end - Host_S_start) / T) - Host_V_start
        A_mean = (Host_V_end - Host_V_start) / T
        Host_A_end = A_mean * 2 - Host_A_start
        Inter_S_start = precessing_train_ver1[:, n:n + 2]
        Inter_V_start = precessing_train_ver1[:, n + 2:n + 4]
        Inter_A_start = precessing_train_ver1[:, n + 4:n + 6]
        Inter_A_end = Inter_A_start
        Inter_V_end = Inter_V_start + T * Inter_A_start
        Inter_S_end = Inter_S_start + Inter_V_start * T + 0.5 * Inter_A_start * T ** 2
        precessing_train_ver1_output = precessing_train_ver1.copy()
        precessing_train_ver1_output[:, 46:48] = Host_V_start
        precessing_train_ver1_output[:, 48:50] = Host_A_start
        precessing_train_ver1_output[:, n + 44:n + 46] = Inter_S_end
        precessing_train_ver1_output[:, n + 46:n + 48] = Inter_V_end
        precessing_train_ver1_output[:, n + 48:n + 50] = Inter_A_end


        t = (generating_lengh - 1) * 0.12
        Host_S_start_test = precessing_test1[:, 0:2]
        Host_V_start_test = precessing_test1[:, 2:4]
        Host_A_start_test = precessing_test1[:, 4:6]

        Host_S_end_test = precessing_test1[:, (2 * n * 16):(2 * n * 16 + 2)]
        Host_V_end_test = (2. * (Host_S_end_test - Host_S_start_test) / t) - Host_V_start_test
        Host_V_end_test = Host_V_start_test + Host_A_start_test * t
        A_mean_test = (Host_V_end_test - Host_V_start_test) / t
        Host_A_end_test = A_mean_test * 2 - Host_A_start_test

        Inter_S_start_test = precessing_test1[:, n:n + 2]
        Inter_V_start_test = precessing_test1[:, n + 2:n + 4]
        Inter_A_start_test = precessing_test1[:, n + 4:n + 6]

        Inter_A_end_test = Inter_A_start_test
        Inter_V_end_test = t * Inter_A_start_test + Inter_V_start_test
        Inter_S_end_test = Inter_S_start_test + Inter_V_start_test * t + 0.5 * Inter_A_end_test * t ** 2

        precessing_test_output1 = precessing_test1.copy()
        precessing_test_output1[:, 2 * n * 16 + 2:2 * n * 16 + 4] = Host_V_start_test
        precessing_test_output1[:, 2 * n * 16 + 4:2 * n * 16 + 6] = Host_A_start_test
        precessing_test_output1[:, 2 * n * 16 + 11 + 0:2 * n * 16 + 11 + 2] = Inter_S_end_test
        precessing_test_output1[:, 2 * n * 16 + 11 + 2:2 * n * 16 + 11 + 4] = Inter_V_end_test
        precessing_test_output1[:, 2 * n * 16 + 11 + 4:2 * n * 16 + 11 + 6] = Inter_A_end_test

        precessing_train_ver_output = np.reshape(precessing_train_ver1_output.T, (n * 2, -1),
                                                 order='F').T
        precessing_test_output = np.reshape(precessing_test_output1.T, (n * 2, -1), order='F').T
        precessing_train_ver_output = np.hstack((ID_Data_train_ver, precessing_train_ver_output))
        precessing_test_output = np.hstack((ID_Data_test, precessing_test_output))

        return precessing_train_ver_output,precessing_test_output

    @staticmethod
    def handle_y(y_train, x_train):
        leng_point = y_train.shape[1]
        index_del = []
        for i in range(y_train.shape[0]):
            x_potint = (y_train[i, 0] - x_train[i, 0]) / (x_train[i, leng_point] - x_train[i, 0])
            y_potint = (y_train[i, 1] - x_train[i, 1]) / (x_train[i, leng_point + 1] - x_train[i, 1])
            if np.isinf(x_potint) or np.isnan(x_potint):
                x_potint = 0.5
            if np.isinf(y_potint) or np.isnan(y_potint):
                y_potint = 0.5
            if abs(x_potint) > 5 or abs(y_potint) > 5:
                index_del.append(i)
            y_train[i, 0] = x_potint
            y_train[i, 1] = y_potint
        y_train = np.delete(y_train, index_del, axis=0)
        x_train = np.delete(x_train, index_del, axis=0)
        return y_train, x_train

    @staticmethod
    def normalizing(input_data):
        output = np.zeros((input_data.shape[0], input_data.shape[1]))
        mmin = np.zeros((1, input_data.shape[1]))
        mmax = np.zeros((1, input_data.shape[1]))

        for i in range(input_data.shape[1]):
            mmin[0, i] = np.min(input_data[:, i])
            mmax[0, i] = np.max(input_data[:, i])
            for j in range(input_data.shape[0]):
                output[j, i] = (input_data[j, i] - mmin[0, i]) / (mmax[0, i] - mmin[0, i])

        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if np.isnan(output[i, j]):
                    output[i, j] = 0

        return output, mmin, mmax

    @staticmethod
    def data_expand(input_train, low, high):
        N_test = np.unique(input_train[:, 0])
        sample = []
        index = 1
        for i in range(len(N_test)):
            exter_trajectory = input_train[input_train[:, 0] == N_test[i], :]
            ID_index = np.arange(low, high,
                                 2)
            for j in range(len(ID_index)):
                for z in range(0, exter_trajectory.shape[0] - ID_index[j], ID_index[j]):
                    sample_1 = np.vstack((np.hstack((index, exter_trajectory[z, :])), np.hstack((index,
                                         exter_trajectory[z + (ID_index[j] / 2).astype(np.int32), :])),
                                         np.hstack((index, exter_trajectory[z + ID_index[j], :]))))
                    if len(sample) == 0:
                        sample = sample_1
                    else:
                        sample = np.vstack((sample, sample_1))
                    index += 1
            print(f'已完成{i} ')
        return sample

    def data_expand_test(self, input_test):
        N_test = np.unique(input_test[:, 0])
        sample = []
        index = 1
        for i in range(len(N_test)):
            exter_trajectory = input_test[input_test[:, 0] == N_test[i], :]
            start_point = np.random.randint(self.args.generating_lengh)
            for j in range(start_point, exter_trajectory.shape[0] - self.args.generating_lengh-1,
                           self.args.generating_lengh):
                sample_1 = np.hstack((np.ones((self.args.generating_lengh, 1)) * index,
                        exter_trajectory[j:j + self.args.generating_lengh, :]))
                if len(sample) == 0:
                    sample = sample_1
                else:
                    sample = np.vstack((sample, sample_1))
                index += 1
                print(index)

        return sample
