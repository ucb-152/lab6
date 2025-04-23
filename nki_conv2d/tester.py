import os
import sys
import argparse
import subprocess
import io
import math
import itertools

import numpy as np

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal
from neuronxcc.nki import benchmark

from conv2d import conv2d_nki as conv2d
from conv2d_ref import conv2d_torch
from utils import performance_requirements, basic_params, fleet_params, dtype_tol, params_name, test_case_params
import json


def test_correctness_conv2d_kernel(kernel, basic_fleet=False, full_fleet=False, test_case=None):
    ref_kernel = conv2d_torch

    if basic_fleet:
        parameter_combinations = basic_params
    elif full_fleet:
        parameter_combinations = fleet_params
    elif test_case:
        parameter_combinations = [test_case_params(test_case)]
    else:
        raise ValueError("Please specify either basic_fleet, full_fleet, or test_case.")

    for params in parameter_combinations:
        input_channels, output_channels, filter_size, batch_size, image_dims, dtype = params
        params_name_str = params_name(params)

        X = np.random.rand(batch_size, input_channels, image_dims[0], image_dims[1]).astype(dtype)
        W = np.random.rand(output_channels, input_channels, filter_size, filter_size).astype(dtype)
        bias = np.random.rand(output_channels).astype(dtype)

        args = [X, W, bias]

        print(f"Running correctness test {params_name_str} -- ", end="", flush=True)

        test_result = kernel(*args)
        ref_results = ref_kernel(*args)

        rtol = dtype_tol[dtype]["rtol"]
        atol = dtype_tol[dtype]["atol"]

        if not np.allclose(ref_results, test_result, rtol=rtol, atol=atol):
            print(f"Failed, writing to file...")

            file_head = params_name(params)

            # Create output directory if it doesn't exist
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)

            # Write reference output to a text file
            ref_output_file = os.path.join(output_dir, f"{file_head}_ref.txt")
            with open(ref_output_file, "w") as ref_file:
                for img_idx in range(ref_results.shape[0]):  # Iterate over images
                    ref_file.write(f"*** Image {img_idx}:\n")
                    for channel_idx in range(ref_results.shape[1]):  # Iterate over output channels
                        ref_file.write(f"** Output Channel {channel_idx}:\n")
                        np.savetxt(ref_file, ref_results[img_idx, channel_idx], fmt="%.4f")
                        ref_file.write("\n")

            # Write test output to a text file
            test_output_file = os.path.join(output_dir, f"{file_head}.txt")
            with open(test_output_file, "w") as test_file:
                for img_idx in range(test_result.shape[0]):  # Iterate over images
                    test_file.write(f"*** Image {img_idx}:\n")
                    for channel_idx in range(test_result.shape[1]):  # Iterate over output channels
                        test_file.write(f"** Output Channel {channel_idx}:\n")
                        np.savetxt(test_file, test_result[img_idx, channel_idx], fmt="%.4f")
                        test_file.write("\n")

            return False
        else:
            print("Passed!")

    if full_fleet:
        print("All correctness tests passed!\n")
    else:
        print("Basic correctness test passed!\n")

    return True


def test_performance_conv2d_kernel(kernel, basic_fleet=False, full_fleet=False, test_case=None, profile=False, record=True):
    if basic_fleet:
        parameter_combinations = basic_params
    elif full_fleet:
        parameter_combinations = fleet_params
    elif test_case:
        parameter_combinations = [test_case_params(test_case)]
    else:
        raise ValueError("Please specify either basic_fleet, full_fleet, or test_case.")

    execution_times = {}

    for params in parameter_combinations:
        input_channels, output_channels, filter_size, batch_size, image_dims, dtype = params
        params_name_str = params_name(params)

        X = np.random.rand(batch_size, input_channels, image_dims[0], image_dims[1]).astype(dtype)
        W = np.random.rand(output_channels, input_channels, filter_size, filter_size).astype(dtype)
        bias = np.random.rand(output_channels).astype(dtype)

        args = [X, W, bias]

        print(f"Benchmarking {params_name_str} test...")

        if profile:
            os.makedirs("profiles", exist_ok=True)
            neff_file_name = params_name(params) + ".neff"
            bench_func = nki.benchmark(
                warmup=20, iters=100, save_neff_name=neff_file_name
            )(kernel)
            bench_func(*args)
            subprocess.run(["mv", neff_file_name, os.path.join("profiles", neff_file_name)], check=True)
        else:
            bench_func = nki.benchmark(
                warmup=20, iters=100
            )(kernel)
            bench_func(*args)
        
        exec_time = bench_func.benchmark_result.nc_latency.get_latency_percentile(99)
        performance_requirement = performance_requirements[params_name_str]
        if exec_time > performance_requirement:
            print(
                f"Performance requirement not met: need to be under {performance_requirement} μs "
            )
            return False
        else:
            print(f"Passed! Executed in {exec_time} μs\n")
        execution_times[params_name_str] = exec_time

    if record:
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        exec_time_file = os.path.join(output_dir, "execution_times.json")
        with open(exec_time_file, "w") as f:
            json.dump(execution_times, f, indent=4)

    return True

def simulate_kernel_wrapper(kernel):
    def temp_func(*args, **kwargs):
        return nki.simulate_kernel(kernel, *args, **kwargs)

    return temp_func

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Generate .neff and .ntff files in profiles directory for neuron-profile usage",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use nki.simulate_kernel to run kernel faster for correctness tests",
    )
    parser.add_argument(
        "--basic",
        action="store_true",
        help="Only run basic tests for correctness and performance",
    )
    parser.add_argument(
        "--test-case", 
        type=str, default=None,
        help="Specific test case to run, see utils.py for options.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the kernel execution time",
    )
    parser.add_argument(
        "--seed", type=int, default=152, help="Seed for random number generation"
    )

    args = parser.parse_args()
    if args.profile and args.simulate:
        print("Warning: --profile and --simulate are mutually exclusive. --profile will be ignored.")
    if args.basic and args.test_case:
        print("Warning: --basic and --test-case are mutually exclusive. --basic will be ignored.")
    if args.record:
        if args.basic or args.test_case:
            print("Warning: --record should not be used with --basic or --test-case.")

    np.random.seed(args.seed)

    if args.simulate:
        conv2d = simulate_kernel_wrapper(conv2d)

    # Correctness tests
    if args.test_case:
        test_result = test_correctness_conv2d_kernel(conv2d, test_case=args.test_case)
        if test_result == False:
            exit()
    else:
        print("Running basic correctness tests for conv2d kernel...")
        test_result = test_correctness_conv2d_kernel(conv2d, basic_fleet=True)
        if test_result == False:
            exit()

        if not args.basic:
            print("Running full fleet of correctness tests for conv2d kernel...")
            test_result = test_correctness_conv2d_kernel(conv2d, full_fleet=True)
            if test_result == False:
                exit()

    # Performance tests
    if not args.simulate:
        if args.test_case:
            test_result = test_performance_conv2d_kernel(conv2d, test_case=args.test_case, profile=args.profile)
            if test_result == False:
                exit()
        else:
            print("Running basic performance tests for conv2d kernel...")
            test_result = test_performance_conv2d_kernel(conv2d, basic_fleet=True, profile=args.profile)
            if test_result == False:
                exit()

            if not args.basic:
                print("Running full fleet of performance tests for conv2d kernel...")
                test_result = test_performance_conv2d_kernel(conv2d, full_fleet=True, profile=args.profile, record=True)
                if test_result == False:
                    exit()

        if args.profile:
            print("Profiling conv2d kernels...")
            if args.test_case:
                param_list = [test_case_params(args.test_case)]
            elif args.basic:
                param_list = basic_params 
            else:
                param_list = fleet_params

            for params in param_list:
                neff_file_name = params_name(params) + ".neff"
                profile_file_name = params_name(params) + ".ntff"
                print(f"Profiling {neff_file_name}... ", end="", flush=True)

                subprocess.run(
                    [
                        "neuron-profile",
                        "capture",
                        "-n",
                        os.path.join("profiles", neff_file_name),
                        "-s",
                        os.path.join("profiles", profile_file_name),
                    ],
                    check=True,
                )     
                print("Done!")  
            print("All profiles generated successfully!")
