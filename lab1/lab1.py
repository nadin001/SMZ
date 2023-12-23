import numpy as np
import torch
from torch.nn.functional import conv2d as libConv2d
import pytest

from conv2D import conv2D


def my_conv2d(input_tensor, weight_tensor, padding=0, dilation=1, stride=1, groups = 1):
    image_height, image_width  = input_tensor.shape
    weight_height, weight_width  = weight_tensor.shape

    H_out = int((image_height - dilation * (weight_height - 1) - 1 + 2 * padding) / stride) + 1
    W_out = int((image_width - dilation * (weight_width - 1) - 1 + 2 * padding) / stride) + 1

    result = np.zeros((H_out, W_out))

    if padding > 0:
        input_tensor = np.pad(input_tensor, padding, mode='constant')

    for y in range(H_out):
        for x in range(W_out):
            result[y, x] = np.sum(input_tensor[y*stride:y*stride+weight_height, x*stride:x*stride+weight_width] * weight_tensor)
    return result

image = torch.randn(1, 1, 5, 5)
kernel = torch.randn(1, 1, 3, 3)

test1_output1 = torch.from_numpy(conv2D(image[0, 0].numpy(), kernel[0, 0].numpy()))

print("Результат использования нашей функции Convolution2D в тесте 1:")
print(test1_output1)
print("Результат использования функции Conv2d библиотеки PyTorch в тесте 1:")
test1_output2 = libConv2d(image, kernel)
print(test1_output2)

test1_output1 = test1_output1.to(test1_output2.dtype)

torch.allclose(test1_output1, test1_output2)