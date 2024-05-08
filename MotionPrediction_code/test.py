import pandas as pd

excel_path = 'E:/MotionPrediction_code/E_trajectory.xlsx'

csv_path = 'E:/MotionPrediction_code/E_trajectory.csv'

df = pd.read_excel(excel_path)
df.to_csv(csv_path, index=False, encoding='utf-8-sig')

