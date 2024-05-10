import os
import numpy as np
import pandas as pd
import csv
from openpyxl import Workbook

txt = np.loadtxt('../inference/output/results.txt')
txtDF1 = pd.DataFrame(txt)
txtDF1.to_csv('../表格输出csv/results.csv', index=False)
num_rows = len(txtDF1.index)
print("CSV 文件共包含 %d 条数据" % num_rows)

def tianjia(xlsname):
    df = pd.read_csv("../表格输出csv/results.csv", dtype=np.float64)
    df.to_csv("../表格输出csv/1.csv", header=None, index=False)

    df.columns = ["ID", "车型", "坐标左", "坐标上", "坐标右", "坐标下", "中心坐标x", "中心坐标y", "帧号"]  # 添加表头
    # 按ID和车型分组计算数量、计算每种车型的占比、将每个ID的车型替换为占比最高的车型
    df.to_csv("../表格输出csv/1带表头.csv", index=False)  # 保留5的倍数
    cols = df.columns.tolist()
    cols.append(cols.pop(1))
    df = df[cols]
    df_count = df.groupby(['ID', '车型']).size().reset_index(name='数量')
    df_count['占比'] = df_count.groupby('ID')['数量'].apply(lambda x: x / x.sum())
    idx = df_count.groupby(['ID'])['占比'].transform(max) == df_count['占比']
    df_most_freq = df_count[idx]
    df = df.merge(df_most_freq[['ID', '车型']], on='ID', how='left')
    df.rename(columns={'车型_y': '车型'}, inplace=True)
    df.drop(columns='车型_x', inplace=True)
    cols = df.columns.tolist()
    cols.insert(1, cols.pop())
    df = df[cols]
    last_column = df.iloc[:, -1]
    last_column = last_column.loc[last_column % 5 == 0]
    selected_data = df[df['帧号'].isin(last_column)]
    selected_data.to_csv("../表格输出csv/每隔5帧.csv", header=None, index=False)  # 保留5的倍数

    df.sort_values(by=['ID', '帧号'], inplace=True, ascending=True) # 转置

    # 记录每个ID的总数，并将结果放到新列中
    grouped = df.groupby('ID').size().reset_index(name='Total')
    id_totals = {}
    for i, row in grouped.iterrows():
        id_value = row['ID']
        id_total = row['Total']
        id_totals[id_value] = id_total
        first_row_index = df.index[df['ID'] == id_value].min()
        df.at[first_row_index, 'Total'] = id_total
    df.to_csv('../表格输出csv/2.csv', index=False)

if __name__ == "__main__":
    xlsname = "../表格输出csv/results.csv"
    tianjia(xlsname)


