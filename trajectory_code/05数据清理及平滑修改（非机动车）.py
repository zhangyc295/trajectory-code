import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter


# Kalman卡尔曼滤波
def Kalman_traj_smooth(data, process_noise_std, measurement_noise_std):
    data = data.copy()
    observations = data[['中心坐标x', '中心坐标y']].values

    transition_matrix = np.array([[1, 0, 1, 0],
                                  [0, 1, 0, 1],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

    observation_matrix = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0]])

    if isinstance(measurement_noise_std, list):
        observation_covariance = np.diag(measurement_noise_std) ** 2
    else:
        observation_covariance = np.eye(2) * measurement_noise_std ** 1

    if isinstance(process_noise_std, list):
        transition_covariance = np.diag(process_noise_std) ** 2
    else:
        transition_covariance = np.eye(4) * process_noise_std ** 0.0001

    initial_state_mean = [observations[0, 0], observations[0, 1], 0, 0]
    initial_state_covariance = np.eye(4) * 8

    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )

    smoothed_states = np.zeros((len(observations), 4))
    smoothed_states[0, :] = initial_state_mean
    current_covariance = initial_state_covariance  # 初始协方差矩阵

    for i in range(1, len(observations)):
        current_state, current_covariance = kf.filter_update(
            smoothed_states[i - 1], current_covariance, observations[i]
        )

        smoothed_states[i, :] = current_state

    data[['中心坐标x', '中心坐标y']] = smoothed_states[:, :2]
    # Update
    return data


# 读取数据
column_names = ['ID', '车型', '坐标左', '坐标上', '坐标右', '坐标下', '中心坐标x', '中心坐标y', '帧号']
df = data = pd.read_csv(r'E:\数据提取\77-109\output108.csv', header=None, names=column_names,
                        encoding='utf-8')

df.columns = column_names

# 将数据命名为“ID”和“帧号”的两列数据按照从小到大的顺序排序
df.sort_values(by=['ID', '帧号'], inplace=True)

# 删除“车型”数据是“1”且ID总数小于50的数据
id_counts = df[df['车型'] == 1]['ID'].value_counts()
to_remove_ids = id_counts[id_counts < 50].index
df = df[~((df['车型'] == 1) & (df['ID'].isin(to_remove_ids)))]

# 设置欧氏距离阈值
threshold_distance = 15.0


# 定义函数判断ID是否跳变
def is_id_jump(x1, y1, x2, y2):
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance > threshold_distance


# 删除轨迹不连续并且离散的点的整个ID所有数据
to_remove_indices = set()

for id_index, group in df[df['车型'] == 1].groupby('ID'):
    x_values = group['中心坐标x'].values
    y_values = group['中心坐标y'].values

    for i in range(1, len(x_values)):
        if is_id_jump(x_values[i - 1], y_values[i - 1], x_values[i], y_values[i]):
            to_remove_indices.add(id_index)
            break

df = df[~((df['车型'] == 1) & (df['ID'].isin(to_remove_indices)))]

# 删除数据
# 删除ID出现次数过少或过多
# data = data.groupby('ID').filter(lambda x: ((len(x) >= 50) ))
data = data.groupby('ID').filter(lambda x: ((len(x) >= 10) & (len(x) <= 2000)))
# 静止车辆筛选
# 计算每组ID对应的X和Y列的坐标极差值（同在一个范围内视为无人机抖动影响）
grouped = data.groupby('ID')[['中心坐标x', '中心坐标y']].agg(lambda x: max(x) - min(x))
# 删除对应的数据
filtered_ids = grouped[(grouped['中心坐标x'] < 100) & (grouped['中心坐标y'] < 100)].index
data = data[~data['ID'].isin(filtered_ids)]
data['FrameDiff'] = data.groupby('ID')['帧号'].diff()
# 删除缺失数据过多的ID
# 找出缺失帧数大于某一数值所在的组，并将这些组的ID筛选出来（删去）
ids_to_delete = data.loc[data['FrameDiff'] > 100, 'ID'].unique()
# 删除要删除的ID所在的行：
data = data[~data['ID'].isin(ids_to_delete)]
# 删除添加的FrameDiff列：
data = data.drop('FrameDiff', axis=1)
# 删除车辆类型为1(非机动车)
# data = data[data['车型'] != 1]


# 单个面积
# 计算条件值
# condition_values = (data['坐标右'] - data['坐标左']) * (data['坐标下'] - data['坐标上'])
# 找到满足条件的ID
# filtered_ids = data.loc[condition_values < 2000, 'ID'].unique()
# 根据条件筛选数据
# data = data[~data['ID'].isin(filtered_ids)]

"""
# 计算平均面积
# 计算坐标右减去坐标左的差值乘以坐标下减去坐标上的差值的乘积，即面积值
data['value'] = (data['坐标右'] - data['坐标左']) * (data['坐标下'] - data['坐标上'])
id_statistics = data.groupby('ID')['value'].agg(['count', 'mean'])
id_statistics.columns = ['count', 'avg_value']
# 删除平均面积满足条件的ID
filtered_ids = id_statistics[id_statistics['avg_value'] < 1000].index
data = data[~data['ID'].isin(filtered_ids)]
"""

# 补全缺失数据
data = data.sort_values(by=['ID', '帧号'])
# 遍历每个ID的数据
for _, group_data in data.groupby('ID'):
    # 获取当前ID的帧号序列
    frame_numbers = group_data['帧号'].values
    # 找出缺失的帧号
    missing_frames = np.setdiff1d(np.arange(frame_numbers.min(), frame_numbers.max() + 1), frame_numbers)
    # 对缺失的帧号进行处理
    for missing_frame in missing_frames:
        # 找到缺失帧号前后两行
        prev_frame_data = group_data.loc[group_data['帧号'] < missing_frame].iloc[-1]
        next_frame_data = group_data.loc[group_data['帧号'] > missing_frame].iloc[0]
        # 等差数列
        interpolation_data = prev_frame_data + (next_frame_data - prev_frame_data) * (
                (missing_frame - prev_frame_data['帧号']) / (next_frame_data['帧号'] - prev_frame_data['帧号'])
        )
        # 将帧号插入缺失位置
        interpolation_data['帧号'] = missing_frame
        # 插入到数据
        data = pd.concat([data, interpolation_data.to_frame().T], ignore_index=True)

grouped_data = data.groupby('ID')
plt.figure(figsize=(8, 8))
smoothed_data_combined = pd.concat([Kalman_traj_smooth(group_data,
                                                       process_noise_std=0.001,
                                                       measurement_noise_std=1)
                                    for _, group_data in grouped_data], sort=False)

# 输出
sorted_output_file_path = r'E:\数据提取\77-109\NEW108.csv'
smoothed_data_combined = smoothed_data_combined.sort_values(by=['帧号', 'ID'])
# smoothed_data_combined = smoothed_data_combined.drop(columns=['value'])  # 删除多余的列
smoothed_data_combined.to_csv(sorted_output_file_path, index=False, header=False, float_format='%.1f')
print("已完成处理")
