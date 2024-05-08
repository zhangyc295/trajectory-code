

import numpy as np
import os
from keras.models import load_model
from scipy import interpolate

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def cubic_hermite_interpolation(valid_frame_labe,valid_x_center,invalid_frame_labe):
    f = interpolate.interp1d(valid_frame_labe, valid_x_center, kind=3)
    return f(invalid_frame_labe)

def trajectory_generation(args, x_input_all):
    n_feature = 14
    n_input = 2
    x_input_all = np.array(x_input_all).reshape(-1, (n_feature * n_input))
    if args.generating_lengh == 25:
        generate_level = 4
    elif args.generating_lengh == 17:
        generate_level = 3
    else:
        print('Trajectory generation error')
        generate_level = []

    data_dir1 = os.path.join(args.data_dir, "Traj_Gen_model_1")
    data_dir2 = os.path.join(args.data_dir, "Traj_Gen_model_2")
    data_dir3 = os.path.join(args.data_dir, "Traj_Gen_model_3")

    if generate_level > 3:
        tg_model_3 = load_model(os.path.join(data_dir3, "trajectory_generation_Farterm（20221207）.h5"))

    tg_model_2 = load_model(os.path.join(data_dir2, "trajectory_generation_longterm（20221101）.h5"))

    tg_model_1 = load_model(os.path.join(data_dir1, "trajectory_generation_shortterm（20221101）.h5"))

    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

    output_all = []
    for i in range(x_input_all.shape[0]):
        x_input = x_input_all[i]
        test_x = x_input.reshape(1, (n_input*n_feature))
        y_pred = []
        PRED_SIZE = args.generating_lengh
        PRED_BATCH = 1
        y_batch_r = np.zeros((PRED_SIZE, n_feature))
        y_pred_final = np.zeros(((PRED_SIZE+n_input), n_feature))

        based_x = test_x.reshape(-1, n_feature)
        for level in range(0, generate_level-1):
            output_j = based_x[0, :]
            for j in range(0, based_x.shape[0]-1, PRED_BATCH):
                input_point = based_x[j:j+2, :].reshape(1, (n_input*n_feature))
                if level == 0 and generate_level > 3:
                    y_batch = tg_model_2.predict(input_point)
                elif level == 1:
                    y_batch = tg_model_2.predict(input_point)
                else:
                    y_batch = tg_model_1.predict(input_point)
                y_batch = y_batch[0, :]
                y_batch[0] = input_point[0, 0] + y_batch[0] * (input_point[0, n_feature] - input_point[0, 0])
                y_batch[1] = input_point[0, 1] + y_batch[1] * (input_point[0, n_feature + 1] - input_point[0, 1])
                end_point = based_x[j+1, :]
                output_j = np.row_stack((output_j, [y_batch, end_point]))
            based_x = output_j

        all_frame = range(0, args.generating_lengh)
        if args.generating_lengh == 17:
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
            Y_all.append(Key_Ponts[frame, 1])
            if (generate_level-1) == 3:
                X_all.append(Interpolated_x[frame * 2 + 0])
                X_all.append(Interpolated_x[frame * 2 + 1])
                Y_all.append(Interpolated_y[frame * 2 + 0])
                Y_all.append(Interpolated_y[frame * 2 + 1])
            if (generate_level-1) == 2:
                X_all.append(Interpolated_x[frame * 3 + 0])
                X_all.append(Interpolated_x[frame * 3 + 1])
                X_all.append(Interpolated_x[frame * 3 + 2])
                Y_all.append(Interpolated_y[frame * 3 + 0])
                Y_all.append(Interpolated_y[frame * 3 + 1])
                Y_all.append(Interpolated_y[frame * 3 + 2])
        X_all.append(Key_Ponts[len(valid_frame_labe)-1, 0])
        Y_all.append(Key_Ponts[len(valid_frame_labe)-1, 1])

        bx = np.array(X_all)
        by = np.array(Y_all)

        output_j = np.array([bx, by]).T
        output_j = np.concatenate((output_j, np.zeros((args.generating_lengh, n_feature-2))), axis=1)
        output_j = output_j.reshape(1, -1)

        if output_all == []:
            output_all = output_j
        else:
            output_all = np.concatenate((output_all, output_j), axis=0)

    return output_all

