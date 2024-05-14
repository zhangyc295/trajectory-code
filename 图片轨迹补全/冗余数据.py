

import csv

# 定义列名
column_names = ['ID', '坐标左', '坐标上', '坐标右', '坐标下', '中心坐标x', '中心坐标y', '帧号']

# 初始化一个字典，用于存储每个ID对应的帧号及其数据行
id_to_frame_data = {}

# 读取CSV文件
filename = r'D:\Users\HP\Desktop\ade\3\data3手动标注.csv'  # CSV文件名
with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=column_names)

    for row in reader:
        # 提取ID和帧号
        id_value = row['ID']
        frame_value = row['帧号']

        # 如果该ID和帧号的组合已经存在，则说明是重复数据
        if (id_value, frame_value) in id_to_frame_data:
            # 将重复的数据行添加到列表中
            id_to_frame_data[(id_value, frame_value)].append(row)
        else:
            # 如果是第一次出现，则初始化为包含当前行的列表
            id_to_frame_data[(id_value, frame_value)] = [row]

# 遍历字典，找出重复的帧号数据并输出
for (id_value, frame_value), rows in id_to_frame_data.items():
    if len(rows) > 1:  # 检查是否有重复的帧号
        print(f"ID: {id_value}, 帧号: {frame_value}, 重复的行数: {len(rows)}")
        for row in rows:
            print(row)


