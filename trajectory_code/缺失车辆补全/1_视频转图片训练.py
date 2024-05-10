import cv2
import os


def get_max_image_number(folder):
    max_number = 0

    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            # 提取文件名中的数字部分
            number = int(os.path.splitext(filename)[0])
            if number > max_number:
                max_number = number

    return max_number


def extract_frames(video_path, output_folder):
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    image_count = 1 + get_max_image_number(output_folder)  # 图片编号起始值
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    while cap.isOpened():
        # 读取视频帧
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # 每隔5帧提取一张图片
        if frame_count % 5 == 0:
            # 设置图片文件名
            filename = f"{image_count}.jpg"
            output_path = os.path.join(output_folder, filename)

            # 保存图片
            cv2.imwrite(output_path, frame)

            image_count += 1  # 图片编号递增

    # 释放视频对象
    cap.release()


# 调用函数进行提取
video_path = r"E:\数据提取\最新视频\N66.mp4"  # 替换为实际的视频文件路径
output_folder = r'E:\数据提取\图片\abc'  # 替换为实际的输出文件夹路径
extract_frames(video_path, output_folder)
