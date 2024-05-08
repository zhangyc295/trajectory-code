import numpy as np
import PredictionEvaluation
from scipy import interpolate
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


class Selection:
    def __init__(self, test_id, args, max_boundary, min_boundary, e_trajectory, extract_one):
        self.test_id = test_id
        self.args = args
        self.max_boundary = max_boundary
        self.min_boundary = min_boundary
        self.E_trajectory = e_trajectory
        self.extract_one = extract_one

    def exhaustion_method(self, perception_time, pos_points_all):
        pos_points_all = np.array(pos_points_all)
        pos_points_all = np.delete(pos_points_all,
                                   np.where((pos_points_all[:, 0] == 0) & (pos_points_all[:, 1] == 0)),
                                   axis=0)
        OPT_th = 1
        index_Truth = \
            np.where((self.E_trajectory[:, 0] == self.extract_one[0, 0]) & (self.E_trajectory[:, 1] == self.extract_one[0, 1]))[0][0]
        Truth_trj = self.E_trajectory[index_Truth:index_Truth + perception_time, 3:5]
        Truth_bef_trj = self.E_trajectory[index_Truth-1, 3:5]
        const_output_all = []
        if self.extract_one[0, 13] == 1:
            k4 = 0.5
        else:
            k4 = 0.5
        fit_dis = []
        possible_points_in = []
        const_output_all = []
        for i in range(pos_points_all.shape[0]):
            Potential_trajectory_allData = np.reshape(pos_points_all[i, 2:], (perception_time, -1))
            Potential_trajectory_allData_output = self.anti_norm_new(Potential_trajectory_allData)
            Potential_trajectory = Potential_trajectory_allData_output[:, :2]
            Potential_trajectory = self.change_point(Potential_trajectory, np.reshape(Truth_bef_trj, (-1, 2)), 10)
            fit_dis.append(PredictionEvaluation.ade_calculation(Truth_trj, Potential_trajectory))
            Potential_trajectory_1D = Potential_trajectory.flatten()
            output_dis, output_risk, sum_dis_path_1, all_step_moving_dis, original_position, const = \
                self.trajectory_benefit_calculating(Potential_trajectory)
            const_output_all.append(np.sign(const))
            possible_points_in.append(
                np.concatenate([pos_points_all[i, :2], [output_dis], [np.sum(np.sign(const))], Potential_trajectory_1D,
                                [output_risk], [sum_dis_path_1]]))
        const_output_all = np.array(const_output_all)
        possible_points_in = np.array(possible_points_in)
        sum_dis_path = possible_points_in[:, possible_points_in.shape[1]-1]
        sum_dis_path = (sum_dis_path - np.min(sum_dis_path)) / (np.max(sum_dis_path) - np.min(sum_dis_path)) + 0.1
        possible_points_in = np.delete(possible_points_in, -1, axis=1)
        possible_points_in[:, -1] = ((possible_points_in[:, -1] - np.min(possible_points_in[:, -1])) /
                                            (np.max(possible_points_in[:, -1]) - np.min(possible_points_in[:, -1])))
        dis_allSample = ((-possible_points_in[:, 2] - (- np.min(-possible_points_in[:, 2]))) / (
                                 np.max(-possible_points_in[:, 2]) - np.min(-possible_points_in[:, 2])))
        dis_allSample = -dis_allSample
        possible_points_in[:, 2] = (1 - k4) * dis_allSample + k4 * possible_points_in[:, -1]
        possible_points_in_fit_dis = np.column_stack(
            (possible_points_in[:, 0:2], fit_dis, np.ones((possible_points_in.shape[0], 1)) * -5,
             possible_points_in[:, 4:possible_points_in.shape[1]]))
        possible_points_allData = possible_points_in[possible_points_in[:, 2].argsort()[::1]]
        aa = possible_points_in[:, :possible_points_in.shape[1]-1]
        optimal_solution = aa[np.argsort(aa[:, 2])]
        optimal_solution_dis = possible_points_in_fit_dis[:, :possible_points_in_fit_dis.shape[1] - 1][
            possible_points_in_fit_dis[:, 2].argsort()[::1]]
        optimal_solution_dis = optimal_solution_dis[optimal_solution_dis[:, 2].argsort()]
        exit_flag = 0

        optimal_solution_5 = optimal_solution[optimal_solution[:, 3] == -5, :]
        optimal_solution_3 = optimal_solution[optimal_solution[:, 3] == -3, :]
        optimal_solution_1 = optimal_solution[optimal_solution[:, 3] == -1, :]
        optimal_solution_11 = optimal_solution[optimal_solution[:, 3] == 1, :]
        optimal_solution_33 = optimal_solution[optimal_solution[:, 3] == 3, :]
        optimal_solution_order = np.concatenate(
            (optimal_solution_5, optimal_solution_3, optimal_solution_1, optimal_solution_11, optimal_solution_33),
            axis=0)

        opt = optimal_solution_order[OPT_th - 1, 4:]

        const_output = const_output_all
        id_best_in_opt = np.where((optimal_solution_order[:, 0] == optimal_solution_dis[0, 0])
                                 & (optimal_solution_order[:, 1] == optimal_solution_dis[0, 1]))[0]
        if len(id_best_in_opt) == 0:
            id_best_in_opt = 0

        id_best_in_dis = np.where((optimal_solution_dis[:, 0] == optimal_solution_order[OPT_th - 1, 0]) & (
                optimal_solution_dis[:, 1] == optimal_solution_order[OPT_th - 1, 1]))[0]

        return opt, exit_flag, possible_points_allData, const_output, id_best_in_opt, id_best_in_dis

    def anti_norm_new(self, norm_data):
        raw_data, _ = self.anti_norm(norm_data, norm_data)
        return raw_data

    def anti_norm(self, tt, yy):
        tt_t = np.zeros((tt.shape[0], tt.shape[1]))
        yy_y = np.zeros((yy.shape[0], yy.shape[1]))
        for i in range(tt.shape[1]):
            for j in range(tt.shape[0]):
                tt_t[j, i] = ((self.max_boundary[0][i] - self.min_boundary[0][i]) * tt[j, i]) + self.min_boundary[0][i]
        for i in range(yy.shape[1]):
            for j in range(yy.shape[0]):
                yy_y[j, i] = ((self.max_boundary[0][i] - self.min_boundary[0][i]) * yy[j, i]) + self.min_boundary[0][i]
        return tt_t, yy_y

    def trajectory_benefit_calculating(self, pon_tra):
        perception_time = pon_tra.shape[0]
        extract = self.extract_one
        extract = np.delete(extract, slice(0, 3), axis=1)
        extract = np.reshape(extract, (-1, 11))
        extract = extract[~np.all(extract == 0, axis=1)]
        extract_pre = self.env_pre(extract, perception_time)
        ori_pos = extract_pre[0:perception_time, 0:2]
        v_a_output = self.v_a_trj_based(pon_tra)
        ori_pos_1 = v_a_output[:, 0:2]
        for i in range(1, ori_pos_1.shape[0]):
            ori_pos_1[i, 0] = ori_pos_1[i - 1, 0] + 0.12 * v_a_output[i, 2]
            ori_pos_1[i, 1] = ori_pos_1[i - 1, 1] + 0.12 * v_a_output[i, 3]
        dis_all = []
        risk_all = []
        dis_path_all = []
        all_ori_risk = []
        all_points = pon_tra
        all_step_moving_dis = np.sqrt(
            (all_points[:, 0] - ori_pos[:, 0]) ** 2 + (all_points[:, 1] - ori_pos[:, 1]) ** 2)
        time_interv = 0.12
        for i in range(all_points.shape[0] - 1):
            extract_time_i = np.reshape(extract_pre[i, :], (-1, 2))
            extract_time_i = np.concatenate((extract_time_i, extract[:, 2:]), axis=1)
            truth_risk = self.risk_cal(all_points[i, :], extract_time_i[1:, :])
            original_risk = self.risk_cal(ori_pos[i, :], extract_time_i[1:, :])
            risk_all.append(np.linalg.norm(truth_risk))
            all_ori_risk.append(np.linalg.norm(original_risk))
            dis = all_points[i + 1, 0] - all_points[i, 0]
            dis_path = PredictionEvaluation.ade_calculation(all_points[i, :], all_points[i + 1, :])
            dis_all.append(dis)
            dis_path_all.append(dis_path)
        dis_all = np.array(dis_all).reshape((-1, 1))
        risk_all = np.array(risk_all).reshape((-1, 1))
        all_ori_risk = np.array(all_ori_risk).reshape((-1, 1))
        output1 = pon_tra
        output2 = (np.mean(risk_all[:, 0]))
        output_dis = (np.sum(dis_all[:, 0]))
        output_risk = (np.mean(risk_all[:, 0]))
        all_step_moving_dis = np.sum(all_step_moving_dis)
        sum_dis_path = np.sum(dis_path_all)
        tra_points = pon_tra

        Risk_constraint = 21
        Cur_constraint = 5
        lin_Angle_constraint = 5
        if self.extract_one[0, 13] == 2:
            a_limit_V = 16.9

        else:
            a_limit_V = 11.4
        speed_constraint = np.max([a_limit_V, self.extract_one[0, 9]])
        acc_cons = 0.4
        lin_Risk = np.max(risk_all) - Risk_constraint
        all_cur = self.curve_calculation(tra_points)
        lin_cur = np.max(np.abs(all_cur)) - Cur_constraint
        distance = 0
        for i in range(tra_points.shape[0] - 1):
            distance += np.linalg.norm(tra_points[i + 1, :] - tra_points[i, :])
        end_speed = (2 * distance / (perception_time * 0.12) * 3.6) - self.extract_one[0, 9]
        lin_speed = end_speed - speed_constraint
        acc = (end_speed - self.extract_one[0, 9]) / 3.6 / (perception_time * 0.12)
        lin_acc = np.abs(acc) - np.abs(acc_cons)
        v_ang = np.array([self.extract_one[0, 5], self.extract_one[0, 6]])
        v_ang = v_ang / np.linalg.norm(v_ang)
        tra_ang = np.array(
            [tra_points[1, 0] - tra_points[0, 0], tra_points[1, 1] - tra_points[0, 1]])
        tra_ang = tra_ang / np.linalg.norm(tra_ang)
        sigma = np.arccos(np.dot(v_ang, tra_ang) / (np.linalg.norm(v_ang) * np.linalg.norm(tra_ang)))
        ang_diff = sigma / np.pi * 180
        if np.isnan(ang_diff):
            ang_diff = 0
        lin_angle_diff = ang_diff - lin_Angle_constraint
        const = np.array([lin_speed, lin_cur, lin_Risk, lin_acc,
                          lin_angle_diff])
        return output_dis, output_risk, sum_dis_path, all_step_moving_dis, ori_pos, const

    @staticmethod
    def env_pre(ext, perception_time):
        time_step = perception_time
        point_index = 0
        ext_time = []
        while point_index < ext.shape[0]:
            vx = [ext[point_index, 2] / (1 / 0.12),
                  ext[point_index, 3] / (1 / 0.12)]
            tra_pre = [ext[point_index, 0], ext[point_index, 1]]
            for i in range(time_step):
                if i == 0:
                    tra_pre = np.vstack((tra_pre, [tra_pre[0] + vx[0], tra_pre[1] + vx[1]]))
                else:
                    tra_pre = np.vstack((tra_pre, [tra_pre[i][0] + vx[0], tra_pre[i][1] + vx[1]]))

            if ext_time == []:
                ext_time = tra_pre
            else:
                ext_time = np.hstack((ext_time, tra_pre))

            point_index += 1
        return ext_time

    @staticmethod
    def v_a_trj_based(trajectory_point):
        v_tra = np.zeros((trajectory_point.shape[0], 2))
        for i in range(1, v_tra.shape[0] - 1):
            for j in range(2):
                v_tra[i, j] = (trajectory_point[i + 1, j] - trajectory_point[i - 1, j]) / (0.12 * 2)
        v_tra[0, :] = v_tra[1, :]
        v_tra[-1, :] = v_tra[-2, :]

        a_tra = np.zeros((trajectory_point.shape[0], 2))
        for i in range(1, v_tra.shape[0] - 1):
            for j in range(2):
                a_tra[i, j] = (v_tra[i + 1, j] - v_tra[i - 1, j]) / (0.12 * 2)
        a_tra[0, :] = a_tra[1, :]
        a_tra[-1, :] = a_tra[-2, :]
        output = np.hstack((trajectory_point, v_tra, a_tra))
        return output

    @staticmethod
    def risk_cal(in_trj, extract):
        risk_max_showing = 10
        g = 0.012
        r_b = 1
        k1 = 0.8
        k3 = 0.05
        char_end = 4
        index_end = 0.2
        char_end_right = 4
        index_end_right = 0.2
        index_int = 2.2
        char_int = 0.1
        int_ben = np.array([[10, 22], [10, 15], [-45, 22], [-45, 15]])
        int_dis = (int_ben[0][1] - in_trj[1])
        if int_dis < 0:
            int_dis = 0

        risk_int = char_int * (int_dis ** index_int) * np.array([0, 1])
        if np.linalg.norm(risk_int) >= risk_max_showing:
            risk_int = risk_max_showing
        end_dis = (in_trj[0] - int_ben[2][0])
        risk_end = char_end * (end_dis ** index_end) * np.array(
            [-1, 0])
        risk_end = np.linalg.norm(risk_end) - np.linalg.norm((char_end * (45 ** index_end) * np.array([-1, 0])))

        if in_trj[1] > int_ben[0][1]:
            risk_end = np.array([0, 0])

        int_ben_rig = [[10, 35], [10, 26], [-45, 35], [-45, 26]]
        int_dis_right = (in_trj[1] - int_ben_rig[1][1])
        if int_dis_right < 0:
            int_dis_right = 0
        risk_int_right = char_int * (int_dis_right ** index_int) * np.array([0, -1])
        if np.linalg.norm(risk_int_right) >= risk_max_showing:
            risk_int_right = risk_max_showing
        end_dis_right = (in_trj[0] - int_ben_rig[2][0])
        risk_end_rig = char_end_right * (end_dis_right ** (-index_end_right)) * np.array([-1, 0])
        risk_end_rig = np.linalg.norm(risk_end_rig) - np.linalg.norm(
            (char_end_right * ((10 + 45) ** (-index_end_right)) * np.array([-1, 0])))

        if in_trj[1] < int_ben_rig[1][1]:
            risk_end_rig = np.array([0, 0])

        risk_inter = 0
        for i in range(extract.shape[0]):
            if np.linalg.norm(extract[i, 2:4]) == 0:
                continue
            inter_direction = -in_trj + extract[i, 0:2]
            if np.linalg.norm(inter_direction) == 0:
                print(in_trj)
                print(extract[i, 0:2])
                print('NaN')
            if extract[i, 10] == 1:
                m_b = 150
            elif extract[i, 10] == 2:
                m_b = 200
            else:
                m_b = 1000

            co = np.dot(extract[i, 2:4], inter_direction) / (
                (np.linalg.norm(extract[i, 2:4]) * np.linalg.norm(inter_direction)))
            if np.isnan(co):
                co = 1
            risk_inter_i = ((g * r_b * m_b) / (np.linalg.norm(inter_direction) ** k1)) * (
                        inter_direction / (np.linalg.norm(inter_direction))) * (
                            np.exp(k3 * extract[i, 6] / 3.6 * (co / np.linalg.norm(co))))
            risk_inter += np.linalg.norm(risk_inter_i)

        if np.linalg.norm(risk_inter) >= risk_max_showing:
            risk_inter = risk_max_showing

        risk = np.linalg.norm(risk_end) + np.linalg.norm(risk_inter) + np.linalg.norm(risk_int) + np.linalg.norm(
                risk_int_right) + np.linalg.norm(risk_end_rig)

        return risk

    @staticmethod
    def curve_calculation(input_tra):
        output_curve = np.zeros((input_tra.shape[0], 1))
        for i in range(1, input_tra.shape[0] - 1):
            line1 = input_tra[i + 1, :] - input_tra[i, :]
            line2 = input_tra[i, :] - input_tra[i - 1, :]

            length1 = np.linalg.norm(line1)
            length2 = np.linalg.norm(line2)
            if length1 == 0 or length2 == 0:

                sigma = np.pi / 2
                angle_diff = 45
            else:
                line1 = line1 / length1
                line2 = line2 / length2
                dot_product = np.dot(line1, line2)
                if dot_product > 1:

                    sigma = 0
                    angle_diff = 0
                elif dot_product < -1:

                    sigma = np.pi
                    angle_diff = 90
                else:
                    sigma = np.arccos(dot_product)
                    angle_diff = sigma / np.pi * 180
            output_curve[i, 0] = angle_diff

        return output_curve


    @staticmethod
    def c_interpolation(valid_frame_labe, valid_x_center, invalid_frame_labe):
        f = interpolate.interp1d(valid_frame_labe, valid_x_center, kind=3)
        return f(invalid_frame_labe)

    def change_point(self, Potential_trajectory, c_v, len_c):
        add_len = 5
        c_v_diff = Potential_trajectory[0, :]-Potential_trajectory[1, :]
        dd = np.zeros((add_len, 2))
        for i in range(0, add_len):
            dd[add_len-i-1, :] = c_v + i * c_v_diff
        np.array(dd)
        Potential_trajectory_all = np.vstack((dd, Potential_trajectory))
        valid_frame_labe = np.hstack(([-5, -4, -3, -2, -1, 0, 1], range(len_c, Potential_trajectory.shape[0])))
        invalid_frame_labe = np.array(range(2, len_c))
        Potential_trajectory_all[2+add_len:len_c+add_len, 0] = self.c_interpolation \
            (valid_frame_labe, np.hstack((Potential_trajectory_all[0:add_len+2, 0],
                                          Potential_trajectory_all[len_c+add_len:, 0])), invalid_frame_labe)
        Potential_trajectory_all[2+add_len:len_c+add_len, 1] = self.c_interpolation \
            (valid_frame_labe, np.hstack((Potential_trajectory_all[0:add_len+2, 1],
                                          Potential_trajectory_all[len_c+add_len:, 1])), invalid_frame_labe)
        Potential_trajectory_all[:, 0] = savgol_filter(Potential_trajectory_all[:, 0], window_length=15, polyorder=4)
        Potential_trajectory_all[:, 1] = savgol_filter(Potential_trajectory_all[:, 1], window_length=15, polyorder=4)

        Potential_trajectory_1 = Potential_trajectory
        return Potential_trajectory

