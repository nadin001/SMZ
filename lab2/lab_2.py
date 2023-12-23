import numpy as np
import torch
from torch.nn.functional import conv3d as libConv3d
import pytest

def conv3D(input_array, kernel_array, stride = 1, padding= 0):
    if input_array.ndim != 3 or kernel_array.ndim != 3:
        raise ValueError("Input and kernel arrays must be 3D with layout 'NHWDC'")
    input_array = np.pad(input_array, padding, mode='constant')
    output_height = (input_array.shape[0] - kernel_array.shape[0]) // stride + 1
    output_width = (input_array.shape[1] - kernel_array.shape[1]) // stride + 1
    output_depth = (input_array.shape[2] - kernel_array.shape[2]) // stride + 1
    output_array = np.zeros((output_height, output_width, output_depth))
    for i in range(output_depth):
        for j in range(output_width):
            for k in range(output_height):
                window = input_array[k:k+kernel_array.shape[0], j:j+kernel_array.shape[1], i:i+kernel_array.shape[2]]
                output_array[k, j, i] = np.sum(window * kernel_array)

    return output_array

input_a = np.random.rand(4, 4, 4)
kernel = np.random.rand(3, 3, 3)
test1_output1 = conv3D(input_a, kernel)
test1_output1 = torch.from_numpy(test1_output1)
print("Результат использования нашей функции Convolution3D в тесте 1:")
print(test1_output1)
print("\nРезультат использования функции Conv3d библиотеки PyTorch в тесте 1:")
#конвертация
input_t = torch.tensor(input_a).unsqueeze(0).unsqueeze(0)
kernel_t = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)
test1_output2 = libConv3d(input_t, kernel_t)
print(test1_output2)

test1_output1 = test1_output1.to(test1_output2.dtype)

torch.allclose(test1_output1, test1_output2)