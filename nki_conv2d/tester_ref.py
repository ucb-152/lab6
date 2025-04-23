import time
import argparse
import os
import itertools

import numpy as np
from conv2d_ref import conv2d_torch, conv2d_numpy, conv2d_numpy_matmul, conv2d_numpy_matmul_tiled
from utils import params_name, dtype_tol

def test_conv2d_ref_kernels(ref_kernel, kernels, benchmark=False):

    # Adjust the input parameters as needed, or add more combinations to test
    # input_channels, output_channels, filter_size, batch_size, image_dims, dtype
    parameter_combinations = [(256, 256, 3, 4, (16, 16), np.float32)]

    for params in parameter_combinations:
        input_channels, output_channels, filter_size, batch_size, image_dims, dtype = params
        params_name_str = params_name(params)

        # Generate random input data
        X = np.random.rand(batch_size, input_channels, image_dims[0], image_dims[1]).astype(dtype)
        W = np.random.rand(output_channels, input_channels, filter_size, filter_size).astype(dtype)
        bias = np.random.rand(output_channels).astype(dtype)
        args = [X, W, bias]

        # Print the test parameters
        print(f"Running {params_name_str} test...")

        # Get the reference result
        ref_results = ref_kernel(*args)

        # Run the target kernels, benchmarking if enabled
        results = []
        for kernel in kernels:
            print(f"- Running kernel {kernel.__name__}... ", end="", flush=True)

            if benchmark:
                start_time = time.time()
                num_iterations = 10
                for _ in range(num_iterations):
                    result = kernel(*args)
                end_time = time.time()
                avg_time = (end_time - start_time) / num_iterations
                print(f"Finished! Average execution time: {avg_time*1e3:.3f} ms")
            else:
                result = kernel(*args)
                print("Finished!")
            results.append(result)

        rtol = dtype_tol[dtype]["rtol"]
        atol = dtype_tol[dtype]["atol"]

        for i, test_result in enumerate(results):
            if not np.allclose(ref_results, test_result, rtol=rtol, atol=atol):
                print(
                    f"- Output mismatch detected between ref kernel {ref_kernel.__name__} and kernel {kernels[i].__name__}:"
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
    test_kernels = [
        conv2d_numpy,
        conv2d_numpy_matmul,
        conv2d_numpy_matmul_tiled
    ]

    test_conv2d_ref_kernels(ref_kernel, test_kernels, benchmark=args.benchmark)
