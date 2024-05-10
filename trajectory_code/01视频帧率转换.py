from moviepy.editor import VideoFileClip
import time
def convert_video_frame_rate(input_video_path,output_video_path,target_frame_rate):
    # 读取输入视频文件
    clip = VideoFileClip(input_video_path)

    # 设置目标帧率
    clip = clip.set_fps(target_frame_rate)

    # 调整分辨率为1k
    clip = clip.resize(height=1080)

    # 保存为输出视频文件
    clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

    print("帧率转换和分辨率调整已完成")


# 调用函数进行帧率转换
input_video_path = r'E:\J85-新马路-浦珠路17：00-17：05-1.MP4'
output_video_path = r'E:\N85-1.MP4'  # 输出视频文件路径
target_frame_rate = 10  # 目标帧率

time1=time.time()
convert_video_frame_rate(input_video_path, output_video_path, target_frame_rate)
time2=time.time()
print(time2-time1)