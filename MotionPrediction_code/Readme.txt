This code mainly includes the following parts:
Main: main program, including the following contents:
1. Data preprocessing: output to E_trajectory.csv
2. Dataset splitting: obtain training, validation, and testing sets required for each model
3. Training
4. Testing part: includes trajectory generation and trajectory selection	
Prediction: testing part, prerequisite: completed training model; including trajectory generation
TrajectorySelection: trajectory selection part, including utility calculation of candidate trajectories
PredictionEvaluation: error calculation of predicted trajectories

This project provides a demonstration demo, including three trained models, testing IDs, and normalization parameters.

This project provides bicycle trajectory data: For the preprocessed E_trajectory.csv, its list represents:
Bicycle ID, Frame ID, Timestamp ID, px, py, vx, vy, ax, ay, v, a, curvature, vehicle type (1 for rb, 2 for eb, 3 for v).

该代码主要包括以下部分：
Main：主程序，包括以下内容：
1. 数据预处理：输出为 E_trajectory.csv
2. 数据集分割：获取每个模型所需的训练集、验证集和测试集
3. 训练
4. 测试部分：包括轨迹生成和轨迹选择
预测：测试部分，前提条件：完成训练模型；包括轨迹生成
轨迹选择：轨迹选择部分，包括候选轨迹的效用计算
预测评估：预测轨迹的误差计算
本项目提供了一个示范演示，包括三个训练有素的模型、测试 ID 和归一化参数。
本项目提供自行车轨迹数据： 对于预处理的 E_trajectory.csv，其 List 表示
自行车 ID、帧 ID、时间戳 ID、px、py、vx、vy、ax、ay、V、a、曲率、车辆类型（1 表示 rb，2 表示 eb，3 表示 v）。