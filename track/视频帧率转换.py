from moviepy.editor import VideoFileClip

def convert_video_frame_rate(input_video_path, output_video_path, target_frame_rate):
    # 读取输入视频文件
    clip = VideoFileClip(input_video_path)

    # 设置目标帧率
    clip = clip.set_fps(target_frame_rate)

    # 保存为输出视频文件
    clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

    print("帧率转换已完成")

# 调用函数进行帧率转换
input_video_path = r'F:\66-82\J66-学森路-守敬路（上为北，17：26）-1.MP4'  # 输入视频文件路径
output_video_path = r'F:\66-82\N66-1.MP4'  # 输出视频文件路径
target_frame_rate = 10  # 目标帧率

convert_video_frame_rate(input_video_path, output_video_path, target_frame_rate)