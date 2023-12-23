from conv3D import conv3D
import torch
import numpy as np
from torch.nn.functional import conv3d as libConv3d
import pytest

def test_1():
    input_a = np.random.rand(4, 4, 4)
    kernel = np.random.rand(3, 3, 3)
    myConv3D = conv3D(input_a, kernel)
    myConv3D = torch.from_numpy(myConv3D)
    input_t = torch.tensor(input_a).unsqueeze(0).unsqueeze(0)
    kernel_t = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)
    torchConv3D = libConv3d(input_t, kernel_t)
    myConv3D = myConv3D.to(torchConv3D.dtype)
    assert torch.allclose(myConv3D, torchConv3D)

def test_2():
    input_a = np.random.rand(8, 8, 8)
    kernel = np.random.rand(4, 4, 4)
    myConv3D = conv3D(input_a, kernel)
    myConv3D = torch.from_numpy(myConv3D)
    input_t = torch.tensor(input_a).unsqueeze(0).unsqueeze(0)
    kernel_t = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)
    torchConv3D = libConv3d(input_t, kernel_t)
    myConv3D = myConv3D.to(torchConv3D.dtype)
    assert torch.allclose(myConv3D, torchConv3D)


def test_3():
    input_a = np.random.rand(6, 6, 6)
    kernel = np.random.rand(1, 1, 1)
    myConv3D = conv3D(input_a, kernel, stride = 1)
    myConv3D = torch.from_numpy(myConv3D)
    input_t = torch.tensor(input_a).unsqueeze(0).unsqueeze(0)
    kernel_t = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)
    torchConv3D = libConv3d(input_t, kernel_t)
    myConv3D = myConv3D.to(torchConv3D.dtype)
    assert torch.allclose(myConv3D, torchConv3D)

def test_4():
    input_a = np.random.rand(10, 10, 10)
    kernel = np.random.rand(5, 5, 5)
    myConv3D = conv3D(input_a, kernel, stride = 1)
    myConv3D = torch.from_numpy(myConv3D)
    input_t = torch.tensor(input_a).unsqueeze(0).unsqueeze(0)
    kernel_t = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)
    torchConv3D = libConv3d(input_t, kernel_t)
    myConv3D = myConv3D.to(torchConv3D.dtype)
    assert torch.allclose(myConv3D, torchConv3D)