def conv2D(input_tensor, weight_tensor, padding=0, dilation=1, stride=1):
    # Get input dimensions
    N, C_in, H_in, W_in = input_tensor.shape
    C_out, C_in_g, K1, K2 = weight_tensor.shape

    # Calculate output dimensions
    H_out = int((H_in + 2 * padding - dilation * (K1 - 1) - 1) / stride + 1)
    W_out = int((W_in + 2 * padding - dilation * (K2 - 1) - 1) / stride + 1)

    # Initialize output tensor
    output_tensor = np.zeros((N, C_out, H_out, W_out))

    # Pad input tensor
    input_padded = np.pad(input_tensor, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    # Loop over batches
    for n in range(N):
        # Loop over output channels
        for c_out in range(C_out):
            # Loop over input channels
            for c_in in range(C_in):
                # Loop over kernel rows
                for i in range(H_out):
                    # Calculate row index in input tensor
                    i_in = i * stride
                    # Loop over kernel columns
                    for j in range(W_out):
                        # Calculate column index in input tensor
                        j_in = j * stride
                        # Extract kernel and input tensor slices
                        kernel = weight_tensor[c_out, c_in, :, :]
                        input_slice = input_padded[n, c_in, i_in:i_in+K1*dilation:dilation, j_in:j_in+K2*dilation:dilation]
                        # Perform convolution
                        output_tensor[n, c_out, i, j] += np.sum(kernel * input_slice)

    return output_tensor