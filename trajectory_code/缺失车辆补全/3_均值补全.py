import csv
import numpy as np
import pandas as pd

"输入上一步生成的1.csv,输出补全后的2.csv，将2.csv的轨迹手动填入之前生成的总的轨迹数据中"

df = pd.read_csv('E:\数据提取/xml/A66.csv', encoding='gbk')
unique_ids = df['ID'].unique()

# 写入CSV文件
with open('E:\数据提取/xml/AAA66.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 写入表头
    header = ['ID', '车型', '坐标左', '坐标上', '坐标右', '坐标下', '中心坐标x', '中心坐标y', '帧号']
    writer.writerow(header)

    for i in unique_ids:
        selected_rows = df[df['ID'] == i]
        first_frame_number = selected_rows['帧号'].iloc[0]
        print(i)
        print(first_frame_number)
        data = selected_rows[['坐标左', '坐标上', '坐标右', '坐标下', '中心坐标x', '中心坐标y']]

        # 扩充数据点的数量
        expanded_data_points = len(data['坐标左']) * 5

        # 创建新的数据字典
        expanded_data = {}

        # 对于每个键（数据列），执行插值操作
        for key in data:
            # 获取当前数据列的值
            values = data[key]

            # 使用线性插值扩充数据
            expanded_values = np.interp(np.linspace(0, 1, expanded_data_points), np.linspace(0, 1, len(values)), values)

            # 保留一位小数
            expanded_values = np.round(expanded_values, 1)

            # 将扩充后的数据存储到新的数据字典中
            expanded_data[key] = expanded_values

        # 添加车型列
        expanded_data['车型'] = [2] * expanded_data_points




        # 生成帧号序列
        frame_numbers = np.arange(first_frame_number, first_frame_number + expanded_data_points)

        # 将帧号列添加到 expanded_data 字典中
        expanded_data['帧号'] = frame_numbers

        # 重新排列列的顺序，将帧号放在最后一列
        reordered_header = header[1:]  # 去掉原始的ID列
        reordered_data = [expanded_data[key] for key in reordered_header]  # 获取重新排列后的数据列
        reordered_header.append(header[-1])  # 将帧号列添加到最后一列

        # 写入数据行
        rows = zip([i] * expanded_data_points, *reordered_data)
        writer.writerows(rows)


print("数据已成功写入CSV文件。")

