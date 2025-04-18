import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal
from neuronxcc.nki import benchmark

from conv2d import conv2d_nki as conv2d

from conv2d_ref import conv2d_torch
import logging
import argparse
import io
import sys

logging.disable(logging.OFF)

def test_correctness_conv2d_kernel(
    kernel,
    dtype=np.float32,
    use_larger_images=False,
):
    ref_impl = conv2d_torch

    input_channels_list = [128, 256]
    output_channels_list = [128, 256]
    kernel_size_list = [3]
    batch_size_list = [4]
    image_dims_list = [(32, 16)]

    if use_larger_images:
        input_channels_list = [256]
        output_channels_list = [256]
        image_dims_list = [(224, 224)]

    for input_channels in input_channels_list:
        for output_channels in output_channels_list:
            for kernel_size in kernel_size_list:
                for batch_size in batch_size_list:
                    for image_dims in image_dims_list:
                        X = np.random.rand(
                            batch_size, input_channels, image_dims[0], image_dims[1]
                        ).astype(dtype)
                        W = np.random.rand(
                            output_channels, input_channels, kernel_size, kernel_size
                        ).astype(dtype)
                        bias = np.random.rand(output_channels).astype(dtype)

                        args = [X, W, bias]

                        out = kernel(*args)
                        out_ref = ref_impl(*args)

                        if not np.allclose(out, out_ref):
                            print(
                                f"Output mismatch detected:\n"
                                f"\tInput Channels: {input_channels}\n"
                                f"\tOutput Channels: {output_channels}\n"
                                f"\tKernel Size: {kernel_size}\n"
                                f"\tBatch Size: {batch_size}\n"
                                f"\tImage Dimensions: {image_dims}\n"
                                f"\tData Type: {dtype}"
                            )

                            return False

    return True

def test_performance_conv2d_kernel(
    kernel,
    dtype=np.float32,
    batch_size=1,
    in_channels=256,
    out_channels=256,
    image_height=224,
    image_width=224,
    kernel_height=3,
    kernel_width=3,
):

    performance_requirements_by_dtype = {
        np.float32: 4300,
        np.float16: 1300
    }

    X = np.random.rand(batch_size, in_channels, image_height, image_width).astype(dtype)
    W = np.random.rand(out_channels, in_channels, kernel_height, kernel_width).astype(dtype)
    bias = np.random.rand(out_channels).astype(dtype)

    args = [X, W, bias]

    dtype_to_str = {
        np.float32: "float32",
        np.float16: "float16"
    }

    if dtype not in dtype_to_str:
        raise ValueError(f"Unsupported dtype {dtype}")
    
    dtype_str = dtype_to_str[dtype]

    bench_func = nki.benchmark(
        warmup=5, iters=20, save_neff_name=f"file_{dtype_str}.neff"
    )(kernel)
    text_trap = io.StringIO()
    sys.stdout = text_trap
    bench_func(*args)
    sys.stdout = sys.__stdout__
    p99_us_student = bench_func.benchmark_result.nc_latency.get_latency_percentile(99)
    print(f"\n\nExecution Time for student implementation: {p99_us_student} μs")

    if p99_us_student > performance_requirements_by_dtype[dtype]:
        print(f"Performance requirement not met: need to be under {performance_requirements_by_dtype[dtype]} μs")
        return False

    return True

def simulate_kernel_wrapper(kernel):
    def temp_func(*args, **kwargs):
        return nki.simulate_kernel(kernel, *args, **kwargs)

    return temp_func

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile", type=str, default=None, help="File to save the neff file"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use nki.simulate_kernel to run student implementation",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random number generation"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.simulate:
        conv2d = simulate_kernel_wrapper(conv2d)
    # running correctness tests
    print(
        "Running correctness test for conv2d kernel with smaller images...",
        end="",
        flush=True,
    )
    test_result = test_correctness_conv2d_kernel(conv2d, use_larger_images=False)
    if test_result:
        print("Passed!")
    else:
        print("Failed :(")

    print(
        "Running correctness test for conv2d kernel with larger images...",
        end="",
        flush=True,
    )
    test_result = test_correctness_conv2d_kernel(conv2d, use_larger_images=True)
    if test_result:
        print("Passed!")
    else:
        print("Failed :(")

    if not args.simulate:
        print("Comparing performance with reference kernel (float32)...")
        test_result = test_performance_conv2d_kernel(conv2d, dtype=np.float32)
        if test_result:
            print("Performance test passed!")
        else:
            print("Performance test failed :(")

        if args.profile is not None:
            save_trace(args.profile + "_float32", "nki_conv2d_float32.neff")
        
        print("Comparing performance with reference kernel (float16)...")
        test_result = test_performance_conv2d_kernel(conv2d, dtype=np.float16)
        if test_result:
            print("Performance test passed!")
        else:
            print("Performance test failed :(")

        if args.profile is not None:
            save_trace(args.profile + "_float16", "nki_conv2d_float16.neff")
