from conv2D import conv2D
import torch
from torch.nn.functional import conv2d as libConv2d
import pytest


def test_1():
    image = torch.randn(1, 1, 5, 5)
    kernel = torch.randn(1, 1, 3, 3)

    myConv2D = torch.from_numpy(conv2D(image[0, 0].numpy(), kernel[0, 0].numpy()))

    torchConv2D = libConv2d(image, kernel)

    myConv2D = myConv2D.to(torchConv2D.dtype)

    assert torch.allclose(myConv2D, torchConv2D)


def test_2():
    image = torch.randn(1, 1, 4, 4)
    kernel = torch.randn(1, 1, 2, 2)

    myConv2D = torch.from_numpy(conv2D(image[0, 0].numpy(), kernel[0, 0].numpy(), stride=2))

    torchConv2D = libConv2d(image, kernel, stride=2)

    myConv2D = myConv2D.to(torchConv2D.dtype)

    assert torch.allclose(myConv2D, torchConv2D)


def test_3():
    image = torch.randn(1, 1, 6, 6)
    kernel = torch.randn(1, 1, 3, 3)

    myConv2D = torch.from_numpy(conv2D(image[0, 0].numpy(), kernel[0, 0].numpy(), stride=2))

    torchConv2D = libConv2d(image, kernel, stride=2)

    myConv2D = myConv2D.to(torchConv2D.dtype)

    assert torch.allclose(myConv2D, torchConv2D)


def test_4():
    image = torch.randn(1, 1, 8, 8)
    kernel = torch.randn(1, 1, 2, 2)

    myConv2D = torch.from_numpy(conv2D(image[0, 0].numpy(), kernel[0, 0].numpy(), stride=2))

    torchConv2D = libConv2d(image, kernel, stride=2)

    myConv2D = myConv2D.to(torchConv2D.dtype)

    assert torch.allclose(myConv2D, torchConv2D)