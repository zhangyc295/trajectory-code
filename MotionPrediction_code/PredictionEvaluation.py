import numpy as np
import matplotlib.pyplot as plt

def ade_calculation(input_arg1, input_arg2):
    mde = []
    diff = input_arg1 - input_arg2
    diff = np.reshape(diff, [-1, 2])
    for i in range(diff.shape[0]):
            mde.append(np.linalg.norm(diff[i][:]))
    output = np.mean(mde)

    return output


def error_sample(args, e_trajectory, test_result_no):


    ade_all = []
    fde_all = []


    test_result_precessing = np.array(test_result_no)

    for i in range(test_result_precessing.shape[0]):

        condition_1 = e_trajectory[:, 0] == test_result_precessing[i, 1]
        condition_2 = e_trajectory[:, 1] == test_result_precessing[i, 2]
        index_ce = np.where(condition_1 & condition_2)[0]
        Truth_no = e_trajectory[int(index_ce): int(index_ce) + args.generating_lengh, :]
        predicted_trajectory = np.reshape(test_result_precessing[i, 3], (-1, 2))

        predicted_trajectory[0, :2] = Truth_no[0, 3:5]

        trend_points = Truth_no
        output1_part = predicted_trajectory
        ade_all.append(ade_calculation(trend_points[1:trend_points.shape[0], 3:5],
                                       output1_part[1:output1_part.shape[0], :]))
        fde_all.append(ade_calculation(trend_points[trend_points.shape[0] - 1, 3:5],
                                       output1_part[output1_part.shape[0] - 1, :]))

    return np.mean(ade_all), np.mean(fde_all)

