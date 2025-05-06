import os
import sys
import argparse
import subprocess
import io
import math
import itertools
import time
from collections import defaultdict

import numpy as np

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal
from neuronxcc.nki import benchmark

from conv2d import conv2d_nki as conv2d
from conv2d_ref import conv2d_torch
from utils import dtype_tol, params_name, test_case_params, basic_test_cases, fleet_test_cases
import json


def test_correctness_conv2d_kernel(kernel, basic_fleet=False, full_fleet=False, test_case=None, record=False):
    ref_kernel = conv2d_torch

    if basic_fleet:
        test_cases = basic_test_cases
    elif full_fleet:
        test_cases = fleet_test_cases
    elif test_case:
        test_cases = {test_case: fleet_test_cases[test_case]}
    else:
        raise ValueError("Please specify either basic_fleet, full_fleet, or test_case.")

    report_data = defaultdict(dict)

    correctness_passed = True
    for test_case in test_cases.keys():
        params = test_case_params(test_case)
        input_channels, output_channels, filter_size, batch_size, image_dims, dtype = params

        st = time.time()
        X = np.random.rand(batch_size, input_channels, image_dims[0], image_dims[1]).astype(dtype)
        W = np.random.rand(output_channels, input_channels, filter_size, filter_size).astype(dtype)
        bias = np.random.rand(output_channels).astype(dtype)
        et = time.time()
        report_data[test_case]['gen_data'] = et-st

        args = [X, W, bias]

        print(f"Running correctness test {test_case} -- ", end="", flush=True)
        
        st = time.time()
        test_result = kernel(*args)
        et = time.time()
        report_data[test_case]['exec_time'] = et-st

        st = time.time()
        ref_results = ref_kernel(*args)
        et = time.time()
        report_data[test_case]['ref_exec_time'] = et-st

        rtol = dtype_tol[dtype]["rtol"]
        atol = dtype_tol[dtype]["atol"]

        if not np.allclose(ref_results, test_result, rtol=rtol, atol=atol):
            report_data[test_case]['correct'] = False
            correctness_passed = False
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
        else:
            report_data[test_case]['correct'] = True
            print("Passed!")

    if correctness_passed:
        if basic_fleet:
            print("All basic correctness tests passed!\n")
        elif full_fleet:
            print("Full correctness test fleet passed!\n")
        elif test_case:
            print(f"{test_case} correctness test passed!\n")

    if record:
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)

        if basic_fleet:
            report_name = f"basic_correctness_test_report"
        elif full_fleet:
            report_name = f"fleet_correctness_test_report"
        elif test_case:
            report_name = f"{test_case}_correctness_test_report"

        report_file = os.path.join(output_dir, f"{report_name}.json")
        print(f"Writing correctness report to {report_file}")
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=4)

    return correctness_passed

def test_performance_conv2d_kernel(kernel, basic_fleet=False, full_fleet=False, test_case=None, profile=False, record=False):
    if basic_fleet:
        test_cases = basic_test_cases
    elif full_fleet:
        test_cases = fleet_test_cases
    elif test_case:
        test_cases = {test_case: fleet_test_cases[test_case]}
    else:
        raise ValueError("Please specify either basic_fleet, full_fleet, or test_case.")

    report_data = defaultdict(dict)

    performance_passed = True
    for test_case in test_cases.keys():
        params = test_case_params(test_case)
        input_channels, output_channels, filter_size, batch_size, image_dims, dtype = params

        st = time.time()
        X = np.random.rand(batch_size, input_channels, image_dims[0], image_dims[1]).astype(dtype)
        W = np.random.rand(output_channels, input_channels, filter_size, filter_size).astype(dtype)
        bias = np.random.rand(output_channels).astype(dtype)
        et = time.time()
        report_data[test_case]['gen_data'] = et-st

        args = [X, W, bias]

        print(f"Benchmarking {test_case} test...")

        st = time.time()
        if profile:
            os.makedirs("profiles", exist_ok=True)
            neff_file_name = test_case + ".neff"
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
        et = time.time()
        report_data[test_case]['exec_time'] = et-st
        
        exec_time = bench_func.benchmark_result.nc_latency.get_latency_percentile(50)
        performance_requirement = test_cases[test_case]
        if exec_time > performance_requirement:
            print(f"Failed :( Executed in {exec_time} μs")
            print(f"Performance requirement not met: need to be under {performance_requirement} μs")
            report_data[test_case]['met_threshold'] = False
            performance_passed = False
        else:
            print(f"Passed! Executed in {exec_time} μs\n")
            report_data[test_case]['met_threshold'] = True
        report_data[test_case]['benchmark_time'] = exec_time

    if performance_passed:
        if basic_fleet:
            print("All basic performance tests passed!\n")
        elif full_fleet:
            print("Full performance test fleet passed!\n")
        elif test_case:
            print(f"{test_case} performance test passed!\n")

    if record:
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)

        if basic_fleet:
            report_name = f"basic_performance_test_report"
        elif full_fleet:
            report_name = f"fleet_performance_test_report"
        elif test_case:
            report_name = f"{test_case}_performance_test_report"

        report_file = os.path.join(output_dir, f"{report_name}.json")
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=4)

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
        help="Use nki.simulate_kernel for correctness tests",
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
    if args.record and args.simulate:
        print("Warning: --record and --simulate are mutually exclusive. --record will be ignored.")
    if args.basic and args.test_case:
        print("Warning: --basic and --test-case are mutually exclusive. --basic will be ignored.")

    np.random.seed(args.seed)

    if args.simulate:
        conv2d = simulate_kernel_wrapper(conv2d)

    # Correctness tests
    if args.test_case:
        test_result = test_correctness_conv2d_kernel(conv2d, test_case=args.test_case, record=args.record)
    elif args.basic:
        print("Running basic correctness tests for conv2d kernel...")
        test_result = test_correctness_conv2d_kernel(conv2d, basic_fleet=True, record=args.record)
    else:
        print("Running full fleet of correctness tests for conv2d kernel...")
        test_result = test_correctness_conv2d_kernel(conv2d, full_fleet=True, record=args.record)
    

    # Performance tests
    if not args.simulate:
        if test_result == False:
            print("\nSkipping performance test(s) due to failed correctness test(s)")
            exit()

        if args.test_case:
            test_performance_conv2d_kernel(conv2d, test_case=args.test_case, profile=args.profile, record=args.record)
        elif args.basic:
            print("Running basic performance tests for conv2d kernel...")
            test_performance_conv2d_kernel(conv2d, basic_fleet=True, profile=args.profile, record=args.record)
        else:
            print("Running full fleet of performance tests for conv2d kernel...")
            test_performance_conv2d_kernel(conv2d, full_fleet=True, profile=args.profile, record=args.record)

        if args.profile:
            print("Profiling conv2d kernels...")
            if args.test_case:
                test_cases = {args.test_case: fleet_test_cases[args.test_case]}
            elif args.basic:
                test_cases = basic_test_cases
            else:
                test_cases = fleet_test_cases

            for test_case in test_cases:
                neff_file_name = test_case + ".neff"
                profile_file_name = test_case + ".ntff"
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
