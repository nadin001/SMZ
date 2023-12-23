import numpy as np

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