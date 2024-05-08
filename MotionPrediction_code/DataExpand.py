import numpy as np

class DataExpand:
    def __init__(self,  args):
        self.args = args

    def data_expand(self, input_train, low, high):
        N_test = np.unique(input_train[:, 0])
        sample = []
        index = 1
        for i in range(len(N_test)):
            exter_trajectory = input_train[input_train[:, 0] == N_test[i], :]
            ID_index = np.arange(low, high, 2)

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

    def data_expand_test(self, input_test, generating_lengh):
        N_test = np.unique(input_test[:, 0])
        sample = []
        index = 1
        for i in range(len(N_test)):
            exter_trajectory = input_test[input_test[:, 0] == N_test[i], :]
            start_point = np.random.randint(generating_lengh)
            for j in range(start_point, exter_trajectory.shape[0] - generating_lengh-1,
                           generating_lengh):
                sample_1 = np.hstack((np.ones((generating_lengh, 1)) * index,
                                         exter_trajectory[j:j + generating_lengh, :]))

                if len(sample) == 0:
                    sample = sample_1
                else:
                    sample = np.vstack((sample, sample_1))
                index += 1
                print(index)
        return sample
