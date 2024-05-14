import numpy as np
import pandas as pd

# 读取数据
column_names = ['ID','坐标左', '坐标上', '坐标右', '坐标下', '中心坐标x', '中心坐标y', '帧号']
df = pd.read_csv(r'D:\Users\HP\Desktop\ade\3\data3视频检测.csv', header=None, names=column_names, encoding='gbk')

# 将数据命名为“ID”和“帧号”的两列数据按照从小到大的顺序排序
df.sort_values(by=['ID', '帧号'], inplace=True)

# 补全缺失数据
data = df.copy()
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

# 按照ID和帧号重新排序
data.sort_values(by=['ID', '帧号'], inplace=True)

# 输出
data.to_csv(r'D:\Users\HP\Desktop\ade\3\data3sp1.csv', index=False, encoding='utf-8',float_format='%.2f')
print("已完成处理")
