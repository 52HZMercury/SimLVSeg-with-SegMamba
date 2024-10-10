import numpy as np
import torch
import cv2


def video_to_tensor(video_path):


    capture = cv2.VideoCapture(video_path)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)
    v = np.zeros((frame_count, 112, 112, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, video_path))
        #112*112
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # # 缩放为128
        # frame = cv2.resize(frame, (128,128))
        v[count, :, :] = frame

    v = v.transpose(( 3, 1, 2, 0))
    v_tensor = torch.from_numpy(v)
    v_tensor = torch.unsqueeze(v_tensor, 0)
    return v_tensor


    # """
    # 将视频文件转换为 PyTorch Tensor。
    #
    # 参数:
    # video_path (str): 视频文件路径。
    #
    # 返回:
    # torch.Tensor: 包含视频帧的四维 Tensor。
    # """
    # # 打开视频文件
    # cap = cv2.VideoCapture(video_path)
    #
    # # 获取视频帧数和帧尺寸
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #
    # # 初始化存储帧的列表
    # frames = []
    #
    # # 逐帧读取视频
    # for _ in range(frame_count):
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     # 将 BGR 图像转换为 RGB，并转换为 Tensor
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     tensor_frame = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
    #     # 缩放为128
    #     tensor_frame = cv2.resize(tensor_frame, (128, 128))
    #     # 添加到帧列表
    #     frames.append(tensor_frame)
    #
    # # 将所有帧组合成一个四维 Tensor
    # video_tensor = torch.stack(frames)
    #
    # # 释放资源
    # cap.release()
    #
    # return video_tensor
