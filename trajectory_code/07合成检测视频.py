import cv2
import pandas as pd
video_path = r"E:\数据提取\降帧视频\N82-1.mp4"
video = cv2.VideoCapture(video_path)
csv_path = r'E:\数据提取\66-82最终确定\output82.csv'
column_names = ['ID', '车型', '坐标左', '坐标上', '坐标右', '坐标下', '中心坐标x', '中心坐标y', '帧号']
df = pd.read_csv(csv_path, header=None, names=column_names)
#表格自己带有表头 header=None删去，最后一列不读column_names[:-1]
# 获取视频的总帧数
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) 
# 获取原始视频的帧率和尺寸信息
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
# 创建VideoWriter对象
output_path = r"E:\数据提取\最新视频\82.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 编码器类型
output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = video.read()
    if not ret:
        break

    # 获取当前帧的车辆轨迹信息
    frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    frame_data = df[df["帧号"] == frame_number]

    # 绘制检测框
    for index, row in frame_data.iterrows():
        bbox = [int(row["坐标左"]), int(row["坐标上"]), int(row["坐标右"]), int(row["坐标下"])]
        class_label = str(row["ID"])
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, class_label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示带有检测框的视频帧
    cv2.imshow("Frame with Detection", frame)

    # 将帧写入输出视频文件
    output_video.write(frame)

    # 显示进度
    progress = frame_number / total_frames * 100
    print(f"Progress: {progress:.2f}%")

    # 按下 'q' 键退出循环
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 释放资源
video.release()
output_video.release()
cv2.destroyAllWindows()