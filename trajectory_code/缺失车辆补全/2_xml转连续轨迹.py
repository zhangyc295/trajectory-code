import csv
import os
import xml.etree.ElementTree as ET

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []

    frame_number = os.path.splitext(os.path.basename(xml_file))[0]  # 获取帧号
    frame_number = int(''.join(filter(str.isdigit, frame_number))) * 5  # 提取数字并转换为整数

    for object in root.findall('object'):
        name = object.find('name').text
        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        data.append([name, xmin, ymin, xmax, ymax, center_x, center_y, frame_number])

    return data

def write_to_csv(xml_folder, csv_file):
    column_names = ['ID', '坐标左', '坐标上', '坐标右', '坐标下', '中心坐标x', '中心坐标y', '帧号']
    data = []

    for file in os.listdir(xml_folder):
        if file.endswith('.xml'):
            xml_file = os.path.join(xml_folder, file)
            data.extend(parse_xml(xml_file))

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)
        writer.writerows(data)

# 指定XML文件夹路径和CSV文件路径
xml_folder = r'E:\数据提取\xml\66-1'
csv_file = r'E:\数据提取\xml\A66.csv'

# 调用函数生成CSV文件
write_to_csv(xml_folder, csv_file)
