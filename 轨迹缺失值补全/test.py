import csv
import numpy as np
import pandas as pd

# 读取原始CSV文件
df = pd.read_csv(r'D:\Users\HP\Desktop\数字.csv', encoding='gbk')

# 按ID和帧号排序
df = df.sort_values(by=['ID', '帧号'])


# 定义插值函数
def interpolate(prev_row, next_row):
    # 为每个坐标列创建一个线性插值数组，结果保留一位小数
    interpolations = {
        col: np.linspace(prev_row[col], next_row[col], num=next_row['帧号'] - prev_row['帧号']).round(1)
        for col in df.columns[1:-1]  # 排除ID和帧号
    }

    # 计算插值帧号
    frame_numbers = np.arange(prev_row['帧号'] + 1, next_row['帧号']).tolist()

    # 生成插值数据
    interpolated_data = []
    for frame_num in frame_numbers:
        data_row = [prev_row['ID']]
        for col in interpolations:
            data_row.extend([interpolations[col][frame_numbers.index(frame_num)]])
        interpolated_data.append(data_row + [frame_num])

    return interpolated_data


# 写入新的CSV文件
with open(r'D:\Users\HP\Desktop\DATA88.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 写入表头
    header = df.columns.tolist()
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
        writer.writerow(current_row.tolist())

        # 更新last_row为当前行
        last_row = current_row

print("数据已成功写入CSV文件。")