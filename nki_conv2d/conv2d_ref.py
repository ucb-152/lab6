import numpy as np
import torch


"""
Performs a 2D convolution operation using PyTorch's built-in functionality.
Args:
    X (array-like): Input tensor of shape (batch_size, in_channels, input_height, input_width).
    W (array-like): Weight tensor of shape (out_channels, in_channels, filter_height, filter_width).
    bias (array-like): Bias tensor of shape (out_channels).
Returns:
    torch.Tensor: The result of the 2D convolution operation, with shape 
                  (batch_size, out_channels, output_height, output_width).
Note:
    This function uses `torch.nn.functional.conv2d` for the convolution operation.
    For more details, refer to the PyTorch documentation:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
"""
def conv2d_torch(X, W, bias):
    X = torch.tensor(X)
    W = torch.tensor(W)
    bias = torch.tensor(bias)

    conv_out = torch.nn.functional.conv2d(X, W, bias)

    return conv_out
 

"""
Performs a 2D convolution operation using a naive NumPy-based implementation.
Args:
    X (array-like): Input tensor of shape (batch_size, in_channels, input_height, input_width).
    W (array-like): Weight tensor of shape (out_channels, in_channels, filter_height, filter_width).
    bias (array-like): Bias tensor of shape (out_channels).
Returns:
    np.ndarray: The result of the 2D convolution operation, with shape 
                (batch_size, out_channels, output_height, output_width).
Note:
    This is a naive implementation of 2D convolution using nested loops and basic NumPy operations.
    It is not optimized for performance and is intended for reference purposes to illustrate
    the underlying calculations, memory accesses, and looping structure of the convolution operation.
"""
def conv2d_numpy(X, W, bias):
    out = None
    
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, _, filter_height, filter_width = W.shape

    H_out = 1 + (input_height - filter_height)
    W_out = 1 + (input_width - filter_width)

    out = np.zeros((batch_size, out_channels, H_out, W_out))
    for b in range(batch_size):
        for c in range(out_channels):
            for i in range(H_out):
                for j in range(W_out):
                    x_ij = X[b, :, i : i + filter_height, j : j + filter_width]
                    out[b, c, i, j] = np.sum(x_ij * W[c]) + bias[c]

    return out

"""
Performs a 2D convolution operation using matrix multiplication in NumPy.
This is not fully optimized, but it shows how the convolution can be accomplished
via reshaping and matmuls.
"""
def conv2d_numpy_matmul(X, W, bias):
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape

    H_out = input_height - filter_height + 1
    W_out = input_width - filter_width + 1

    # Reshape the filters and images

    X_reshaped = np.swapaxes(X, 0, 1)
    # X: (in_channels, batch_size, input_height, input_width)
    
    W_reshaped = np.swapaxes(W, 0, 1)
    W_reshaped = W_reshaped.reshape( in_channels, out_channels, filter_height * filter_width)
    W_reshaped = np.swapaxes(W_reshaped, 1, 2)
    # W: (in_channels, filter_size = filter_height * filter_width, out_channels)

    out = np.zeros((batch_size, out_channels, H_out, W_out)) 
    # Perform the convolution using matrix multiplication
    for i_c in range(in_channels):
        for i in range(H_out):
            for j in range(W_out):
                # Extract the relevant patch from the input image
                x_ij = X_reshaped[i_c, :, i : i + filter_height, j : j + filter_width]
                # Reshape the patch to match the filter shape
                x_ij = x_ij.reshape(batch_size, filter_height * filter_width)

                w_ij = W_reshaped[i_c]

                out_ij = x_ij @ w_ij
                # out_ij: (batch_size, out_channels)

                out[:, :, i, j] += out_ij
    
    # Add the bias
    out += bias.reshape(1, out_channels, 1, 1)

    return out


"""
Performs a 2D convolution operation using matrix multiplication & tiling in NumPy.
This is not fully optimized, but it shows how the convolution can be accomplished
via reshaping, matmuls, and tiling.
"""
def conv2d_numpy_matmul_tiled(X, W, bias):
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape

    H_out = input_height - filter_height + 1
    W_out = input_width - filter_width + 1

    tile_c = 128
    n_c_in_tiles = in_channels // tile_c
    n_c_out_tiles = out_channels // tile_c

    # Reshape the filters and images for tiling and access

    X_reshaped = np.swapaxes(X, 0, 1)
    X_reshaped = X_reshaped.reshape(n_c_in_tiles, tile_c, batch_size, input_height, input_width)
    # X: (n_c_in_tiles, tile_c, batch_size, input_height, input_width)
    
    W_reshaped = W.reshape(n_c_out_tiles, tile_c, n_c_in_tiles, tile_c, filter_height * filter_width)
    W_reshaped = np.moveaxis(W_reshaped, [0, 1, 2, 3, 4], [3, 4, 0, 1, 2])
    # W: (n_c_in_tiles, tile_c, filter_size = filter_height * filter_width, n_c_out_tiles, tile_c)

    out = np.zeros((batch_size, n_c_out_tiles, tile_c, H_out, W_out)) 
    # Perform the convolution using matrix multiplication

    for c_in_tile_idx in range(n_c_in_tiles):
        x_tile = X_reshaped[c_in_tile_idx]
        w_tile = W_reshaped[c_in_tile_idx]

        for i_c in range(tile_c):
            x_i_c = x_tile[i_c]
            w_i_c = w_tile[i_c]
            for i in range(H_out):
                for j in range(W_out):
                    x_ij = x_i_c[:, i : i + filter_height, j : j + filter_width]
                    x_ij = x_ij.reshape(batch_size, filter_height * filter_width)

                    for c_out_tile_idx in range(n_c_out_tiles):
                        w_ij = w_i_c[:, c_out_tile_idx, :]
                        w_ij = w_ij.reshape(filter_height * filter_width, tile_c)

                        out_ij = x_ij @ w_ij

                        out[:, c_out_tile_idx, :, i, j] += out_ij
    
    # Reshape the output to the desired shape
    out = out.reshape(batch_size, out_channels, H_out, W_out)
    
    # Add the bias
    out += bias.reshape(1, out_channels, 1, 1)

    return out
