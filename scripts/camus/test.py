import os

import numpy as np



data_dir = '/workdir1/cn24/data/CAMUS'
# self.n_frames = n_frames

patients = sorted(
    [filename.split('_')[0] for filename in os.listdir(data_dir) if '_gt.npy' in filename])

patient = patients[18]
a4c_seq = np.load(os.path.join(data_dir, f'{patient}_a4c_seq.npy'))
a4c_gt = np.load(os.path.join(data_dir, f'{patient}_a4c_gt.npy'))

a4c_seq = np.float32(a4c_seq) / 255.
a4c_gt = np.float32(a4c_gt)

if len(a4c_seq.shape) == 3:
    a4c_seq = a4c_seq[..., np.newaxis] * np.ones((1, 1, 1, 3))

a4c_seq = (a4c_seq - 0.12741163) / 0.19557191


import numpy as np

def pad_array_with_origin_images_seq(X,M):
    """
    将输入数组 a4c_seq 的第一维扩展到 128，通过重复后三维数据实现。

    参数:
    a4c_seq (numpy.ndarray): 形状为 (22, 112, 112, 3) 的数组。

    返回:
    numpy.ndarray: 形状为 (128, 112, 112, 3) 的数组。
    """
    # 计算需要重复的次数
    num_repeats = -(-M // X.shape[0])  # 向上取整

    # 重复数组
    repeated_array = np.tile(X, (num_repeats, 1, 1, 1))

    # 截取前 target_length 个元素
    padded_array = repeated_array[:M]

    return padded_array

def pad_array_with_origin_images_gt(X, M):
    """
    将输入数组 a4c_gt 的第一维扩展到指定的目标长度，通过重复后二维数据实现。

    参数:
    a4c_gt (numpy.ndarray): 形状为 (22, 112, 112) 的数组。
    M (int): 目标长度，即扩展后的数组第一维的长度。

    返回:
    numpy.ndarray: 形状为 (M, 112, 112) 的数组。
    """
    # 计算需要重复的次数
    num_repeats = -(-M // X.shape[0])  # 向上取整

    # 重复数组
    repeated_array = np.tile(X, (num_repeats, 1, 1))

    # 截取前 M 个元素
    padded_array = repeated_array[:M]

    return padded_array


print(a4c_seq.shape)
print(a4c_gt.shape)

if 128 > a4c_seq.shape[0]:
    a4c_seq = pad_array_with_origin_images_seq(a4c_seq, 128)
    a4c_gt = pad_array_with_origin_images_gt(a4c_gt, 128)

print(a4c_seq.shape)
print(a4c_gt.shape)
assert a4c_seq.shape[0] == 128



import matplotlib.pyplot as plt

# 假设 a4c_seq 已经加载并处理完毕
# a4c_seq 的形状应为 (128, 112, 112, 3)

# 提取第一帧图像
first_frame = a4c_seq[22]
first_gt = a4c_gt[22]

# 使用 matplotlib 显示图像
plt.imshow(first_frame)
plt.title('22nd Frame of a4c_seq')
plt.axis('off')  # 关闭坐标轴
plt.show()
# 保存图像到当前目录
plt.savefig('22_frame.png')

# 使用 matplotlib 显示图像
plt.imshow(first_gt)
plt.title('22nd Frame of a4c_gt')
plt.axis('off')  # 关闭坐标轴
plt.show()
# 保存图像到当前目录
plt.savefig('22_gt.png')

print(a4c_seq.shape)
print(a4c_gt.shape)
