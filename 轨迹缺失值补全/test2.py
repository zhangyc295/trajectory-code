import csv
import numpy as np
import pandas as pd

# 读取原始CSV文件
df = pd.read_csv(r'D:\Users\HP\Desktop\数字.csv', encoding='gbk')

# 按ID和帧号排序
df = df.sort_values(by=['ID', '帧号'])


# 定义插值函数
def interpolate(prev_row, next_row):
    # 获取需要插值的列
    cols_to_interpolate = ['坐标左', '坐标上', '坐标右', '坐标下', '中心坐标x', '中心坐标y']

    # 为每个坐标列创建一个线性插值数组，结果保留一位小数
    interpolations = {
        col: np.linspace(getattr(prev_row, col), getattr(next_row, col),
                         num=int(np.round(next_row['帧号'] - prev_row['帧号'])) + 1).round(1)
        for col in cols_to_interpolate
    }

    # 计算插值帧号
    frame_numbers = np.arange(prev_row['帧号'] + 1, next_row['帧号'] + 1).tolist()

    # 生成插值数据
    interpolated_data = [(prev_row['ID'],) + tuple(interpolations[col][i] for col in cols_to_interpolate) + (frame_num,)
                         for i, frame_num in enumerate(frame_numbers)]

    return interpolated_data


# 写入新的CSV文件
with open(r'D:\Users\HP\Desktop\DATA88.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 写入表头
    header = ['ID', '坐标左', '坐标上', '坐标右', '坐标下', '中心坐标x', '中心坐标y', '帧号']
    writer.writerow(header)

    # 上一行数据
    last_row = None
    for index, current_row in df.iterrows():
        if last_row is not None and last_row['ID'] == current_row['ID']:
            # 计算缺失的帧号数量
            num_missing_frames = current_row['帧号'] - last_row['帧号'] - 1

            # 如果存在缺失的帧号，则进行插值
            if num_missing_frames > 0:
                # 插值并写入数据
                interpolated_data = interpolate(last_row, current_row)
                for data in interpolated_data:
                    writer.writerow(data)

        # 写入当前行数据
        writer.writerow([getattr(current_row, col) for col in header])

        # 更新last_row为当前行
        last_row = current_row

print("数据已成功写入CSV文件。")