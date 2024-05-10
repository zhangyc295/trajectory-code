import pandas as pd

path = r'E:\数据提取\66-82最终确定\output82新.csv'
output_path = r'E:\数据提取\66-82最终确定\output82.csv'

# 读取CSV文件并按帧号排序
column_names = ['ID', '车型', '坐标左', '坐标上', '坐标右', '坐标下', '中心坐标x', '中心坐标y', '帧号']
data = pd.read_csv(path, header=None, names=column_names)
data = data.sort_values(by='帧号')

# 生成新的ID映射关系
unique_ids = data['ID'].unique()
new_ids = range(1, len(unique_ids) + 1)
id_mapping = dict(zip(unique_ids, new_ids))

# 更新ID列
data['ID'] = data['ID'].replace(id_mapping)

# 写入新的CSV文件
data.to_csv(output_path, index=False, header=False, float_format='%.1f')
