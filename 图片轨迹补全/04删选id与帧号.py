import csv

# 指定CSV文件路径
csv1_path = r'D:\Users\HP\Desktop\ade\3\data3手动标注.csv'  # 文件1.csv的路径
csv2_path = r'D:\Users\HP\Desktop\ade\3\data3视频检测.csv'  # 文件2.csv的路径
output_path = r'D:\Users\HP\Desktop\ade\3\sp.csv'

# 读取文件1.csv并创建索引映射
def create_index_mapping(csv_path):
    index_mapping = {}
    with open(csv_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            key = (row[0], row[-1])  # 使用ID和帧号作为键
            index_mapping[key] = row
    return index_mapping

def filter_csv_by_index_mapping(csv_file, index_mapping):
    with open(output_path, mode='w', encoding='utf-8', newline='') as output_file:
        writer = csv.writer(output_file)
        # 由于我们不写入表头，所以这一行被注释掉
        # writer.writerow(column_names)  # 写入表头
        for row in csv.reader(csv_file):
            key = (row[0], row[-1])  # 使用ID和帧号作为键
            if key in index_mapping:
                writer.writerow(row)  # 写入匹配的行

# 主函数
def main():
    index_mapping = create_index_mapping(csv1_path)
    # 重置文件指针
    with open(csv2_path, mode='r', encoding='utf-8') as csv_file:
        filter_csv_by_index_mapping(csv_file, index_mapping)

if __name__ == '__main__':
    main()
