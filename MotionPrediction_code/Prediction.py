import numpy as np
import Path_Generation_Deeplearing
import TrajectorySelection
import DataExpand
import dill
from scipy import interpolate


class TwoLayerModelTest:
    def __init__(self, resolution, test_id, args, max_boundary, min_boundary, E_trajectory):
        self.resolution = resolution
        self.test_id = test_id
        self.args = args
        self.max_boundary = max_boundary
        self.min_boundary = min_boundary
        self.E_trajectory = E_trajectory

    def test(self):
        where_best = []
        where_opt = []
        where_const = []
        all_exit_flag = []
        test_result_no = []

        for ID in range(self.test_id.shape[0]):
            index = self.test_id[ID, 1]
            index_in = self.test_id[ID, 2]-1
            index_in = int(index_in)
            extract_trajectory = self.E_trajectory[self.E_trajectory[:, 0] == index, :]
            if index_in + self.args.generating_lengh > extract_trajectory.shape[0]:
                continue
            extract_trajectory = extract_trajectory[index_in:index_in + self.args.generating_lengh, :]
            aa = []
            for i in range(1, len(extract_trajectory[:, 3])):
                aa.append((extract_trajectory[:, 3][i] - extract_trajectory[:, 3][i - 1]))
            if np.any(np.array(aa) >= 0):
                continue
            extract_one = extract_trajectory[0, :]
            extract_one = extract_one.reshape(1, -1)

            possible_points, status = self.batch_trajectory_generation(extract_one, 0)
            if status != 0:
                continue
            ts = TrajectorySelection.Selection(self.test_id, self.args, self.max_boundary,
                                               self.min_boundary, self.E_trajectory, extract_one)
            opt, exit_flag, possible_points_all_data, const_output, id_best_in_opt, id_best_in_dis \
                = ts.exhaustion_method(self.args.generating_lengh, possible_points)
            where_best.append(id_best_in_opt)
            where_opt.append(id_best_in_dis)
            where_const.append(const_output)
            all_exit_flag.append(exit_flag)
            test_result_no.append([len(where_best), index, index_in, opt, np.sum(np.sign(const_output))])
        return test_result_no, all_exit_flag, where_opt

    def batch_trajectory_generation(self, extract_one, typ):
        perception_interval = self.resolution / np.sqrt(2)
        v_cos = 15
        if extract_one[0][13] == 2:
            a_limit_border = 0.36
        else:
            a_limit_border = 0.22
        a_limit_border = np.mean([a_limit_border, 1.2 * max(a_limit_border, abs(extract_one[0][7]))])
        if extract_one[0][13] == 2:
            a_limit_horizontal = 0.189
        else:
            a_limit_horizontal = 0.15
        a_limit_horizontal = np.mean([a_limit_horizontal, 1.2 * max(a_limit_horizontal, abs(extract_one[0][8]))])
        v_max = extract_one[0][9] + (self.args.generating_lengh * 0.12) * a_limit_border * 3.6
        border_near = int(np.ceil((min(0, (extract_one[0][5] * (self.args.generating_lengh * 0.12) +
                        0.5 * a_limit_border * ((self.args.generating_lengh * 0.12) ** 2))))
                                 / perception_interval))
        border_far = int(np.ceil((min(0, (extract_one[0][5] * (self.args.generating_lengh * 0.12) -
                        0.5 * a_limit_border * ((self.args.generating_lengh * 0.12) ** 2))))
                                 / perception_interval))
        if v_max > v_cos:
            border_far = int(np.ceil((((-v_cos / 3.6) * (self.args.generating_lengh * 0.12)) / perception_interval)))
        later_dis_1 = (extract_one[0][6] * (self.args.generating_lengh * 0.12) + 0.5 * a_limit_horizontal * (
                    (self.args.generating_lengh * 0.12) ** 2))
        later_dis_0 = (extract_one[0][6] * (self.args.generating_lengh * 0.12) - 0.5 * a_limit_horizontal * (
                    (self.args.generating_lengh * 0.12) ** 2))
        if (extract_one[0][4] + later_dis_1) > 25:
            later_dis_1 = later_dis_1 * 0.7
        if (extract_one[0][4] + later_dis_0) < 22:
            later_dis_0 = later_dis_0 * 0.7
        horizontal_range_1 = int(np.ceil(later_dis_1 / perception_interval))
        horizontal_range_0 = int(np.ceil(later_dis_0 / perception_interval))

        border = [border_near, border_far]
        border = np.sort(border)
        possible_points = []
        index = 1
        x_input_all = []
        for i in range(border[0], border[1]+1):
            for j in range(min(horizontal_range_1, horizontal_range_0),
                           max(horizontal_range_1, horizontal_range_0)+1):

                start_point = extract_one[0][3:5]
                end_point = np.array([i, j]) * perception_interval + start_point
                if typ == 0:
                    end_point += np.random.rand(2,) * self.resolution / 2 - self.resolution / 4
                com_od = self.complete_feature_od_points(start_point, end_point, extract_one)
                complete_data_od_points_output = self.norm_new(com_od)
                possible_points.append([i, j])
                x_input_all.append(complete_data_od_points_output)

                index += 1
        pre_data_test = Path_Generation_Deeplearing.trajectory_generation(self.args, x_input_all)
        if len(pre_data_test) > 0:
            status = 0
        else:
            status = 1
        possible_points = np.concatenate((np.array(possible_points), pre_data_test), axis=1)
        return possible_points, status

    def complete_feature_od_points(self, start_point, end_point, extract_one):
        v0 = extract_one[0][5:7]
        t = (self.args.generating_lengh - 1) * 0.12
        a0 = extract_one[0][7:9]
        a_mean = ((end_point - start_point) - v0 * t) * 2 / (t ** 2)
        vt = (a_mean * t) + v0
        at = a_mean * 2 - a0
        inter_data = extract_one[0][14:25]
        a_inter_data = inter_data[4:6]
        v_inter_data = t * a_inter_data
        p_inter_data = v_inter_data * \
                         t + 0.5 * a_inter_data * t ** 2
        inter = np.hstack((p_inter_data, v_inter_data, a_inter_data))
        x_output = np.hstack((extract_one.flatten()[3:9], extract_one.flatten()[13]))
        x_output = np.hstack((x_output, inter_data[0:6], inter_data[10]))
        x_output = np.hstack((x_output, end_point, vt, at, extract_one.flatten()[13]))
        x_output = np.hstack((x_output, inter, inter_data[10]))
        output = x_output
        return output

    def norm_new(self, com_od):
        nu_len = self.max_boundary.shape[1]
        complete_data_od_points_1 = np.reshape(
            com_od, (-1, nu_len))
        complete_data_od_points_2 = np.zeros_like(complete_data_od_points_1)
        for i in range(complete_data_od_points_1.shape[0]):
            for j in range(nu_len):
                complete_data_od_points_2[i, j] = (complete_data_od_points_1[i, j] - self.min_boundary[0][j]) / \
                                                  (self.max_boundary[0][j] - self.min_boundary[0][j])
        complete_data_od_points_output = np.reshape(
            complete_data_od_points_2, (-1, len(com_od)))
        return complete_data_od_points_output



