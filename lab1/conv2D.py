import numpy as np

def conv2D(input_tensor, weight_tensor, padding=0, dilation=1, stride=1, groups = 1):
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