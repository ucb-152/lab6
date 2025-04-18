import numpy as np
from conv2d_ref import conv2d_torch, conv2d_numpy, conv2d_numpy_matmul, conv2d_numpy_matmul_tiled
import time
import argparse
import os


def test_conv2d_ref_kernels(ref_kernel, kernels, benchmark=False):

    # Adjust the input parameters as needed, or add more combinations to test
    input_channels_list = [256]
    output_channels_list = [256]
    filter_size_list = [3]
    batch_size_list = [4]
    image_dims_list = [(32, 16)]
    dtypes = [np.float32, np.float16]

    # Tolerance for floating point comparisons (rtol, atol)
    dtype_tol = {
        np.float32: {"rtol": 1e-5, "atol": 1e-8},
        np.float16: {"rtol": 1e-2, "atol": 1e-5},
    }

    for input_channels in input_channels_list:
        for output_channels in output_channels_list:
            for filter_size in filter_size_list:
                for batch_size in batch_size_list:
                    for image_dims in image_dims_list:
                        for dtype in dtypes:

                            # Generate random input data
                            X = np.random.rand(batch_size, input_channels, image_dims[0], image_dims[1]).astype(dtype)
                            W = np.random.rand(output_channels, input_channels, filter_size, filter_size).astype(dtype)
                            bias = np.random.rand(output_channels).astype(dtype)
                            args = [X, W, bias]

                            # Get the reference result
                            ref_results = ref_kernel(*args)

                            # Run the target kernels, benchmarking if enabled
                            results = []
                            if benchmark:
                                for kernel in kernels:
                                    start_time = time.time()
                                    num_iterations = 10
                                    for _ in range(num_iterations):
                                        result = kernel(*args)
                                    end_time = time.time()
                                    avg_time = (end_time - start_time) / num_iterations
                                    print(
                                        f"Kernel {kernel.__name__} average execution time: {avg_time*1e3:.3f} ms\n"
                                        f"- Input Channels: {input_channels}, Output Channels: {output_channels}, Filter Size: {filter_size}, " \
                                        f"Batch Size: {batch_size}, Image Dimensions: {image_dims}, Data Type: {dtype}"
                                        )
                                    results.append(result)
                            else:
                                for kernel in kernels:
                                    result = kernel(*args)
                                    results.append(result)

                            rtol = dtype_tol[dtype]["rtol"]
                            atol = dtype_tol[dtype]["atol"]

                            for i, test_result in enumerate(results):
                                if not np.allclose(ref_results, test_result, rtol=rtol, atol=atol):
                                    print(
                                        f"Output mismatch detected between kernel {ref_kernel.__name__} and kernel {kernels[i].__name__}:\n"
                                        f"- Input Channels: {input_channels}, Output Channels: {output_channels}, Kernel Size: {kernel_size}, " \
                                        f"Batch Size: {batch_size}, Image Dimensions: {image_dims}, Data Type: {dtype}"
                                    )
                                    
                                    # Create output directory if it doesn't exist
                                    output_dir = "outputs"
                                    os.makedirs(output_dir, exist_ok=True)

                                    # Write reference output to a text file
                                    ref_output_file = os.path.join(output_dir, f"{ref_kernel.__name__}_output.txt")
                                    with open(ref_output_file, "w") as ref_file:
                                        for img_idx in range(ref_results.shape[0]):  # Iterate over images
                                            ref_file.write(f"*** Image {img_idx}:\n")
                                            for channel_idx in range(ref_results.shape[1]):  # Iterate over output channels
                                                ref_file.write(f"** Output Channel {channel_idx}:\n")
                                                np.savetxt(ref_file, ref_results[img_idx, channel_idx], fmt="%.4f")
                                                ref_file.write("\n")

                                    # Write test output to a text file
                                    test_output_file = os.path.join(output_dir, f"{kernels[i].__name__}_output.txt")
                                    with open(test_output_file, "w") as test_file:
                                        for img_idx in range(test_result.shape[0]):  # Iterate over images
                                            test_file.write(f"*** Image {img_idx}:\n")
                                            for channel_idx in range(test_result.shape[1]):  # Iterate over output channels
                                                test_file.write(f"** Output Channel {channel_idx}:\n")
                                                np.savetxt(test_file, test_result[img_idx, channel_idx], fmt="%.4f")
                                                test_file.write("\n")

                                    print(f"Reference output saved to: {ref_output_file}")
                                    print(f"Test output saved to: {test_output_file}")
                                    
                                    return False
                                
    print("All kernels passed the reference test!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test conv2d reference kernels.")
    parser.add_argument("--benchmark", action="store_true", help="Enable benchmarking of kernels.")
    args = parser.parse_args()

    # Kernel to use as a golden model reference
    ref_kernel = conv2d_torch

    # Kernels to verify against ref_kernel
    # When --benchmark is set, each kernel below will be benchmarked
    kernels = [
        conv2d_torch,
        conv2d_numpy,
        conv2d_numpy_matmul,
        conv2d_numpy_matmul_tiled
    ]

    test_conv2d_ref_kernels(ref_kernel, kernels, benchmark=args.benchmark)
