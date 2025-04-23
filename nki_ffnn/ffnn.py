import os
import numpy as np

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt

from ffnn_ref import NeuralNetwork
from kernels import nki_forward, nki_predict
from utils import load_data, load_results, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
import argparse

# Enable .neff output for profiling
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"]= " --disable-dge "


def benchmark_nki(nki_func, *args, **kwargs):
    bench_func = nki.benchmark(warmup=5, iters=10)(nki_func)
    bench_func(*args, **kwargs)
    latency_res = bench_func.benchmark_result.nc_latency
    exec_time = latency_res.get_latency_percentile(50)
    print("Execution Time: {:.2f} ms ".format(exec_time / 1000.0))

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Feedforward Neural Network")
    parser.add_argument("-d", "--data_path", default="ffnn", help="Path to the weights, biases, input, and result .bin files")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Run benchmarking on available matmul kernels")
    args = parser.parse_args()

    # Update the data path for load_data function
    data_path = args.data_path

    # Load input data
    X, W1, b1, W2, b2 = load_data(path=data_path)

    # Run the Feedforward Neural Network on the inputs
    predictions = nki_predict(X, W1, b1, W2, b2)

    # Print the predictions
    print("Predictions:")
    print(predictions)

    # Compare against golden model
    Y = load_results(path=data_path)
    if np.array_equal(Y, predictions):
        print("Predictions match the golden model.")
    else:
        print("Predictions do not match the golden model.")
        print(Y.shape)
        print(predictions.shape)
        mismatched_indices = np.where(Y != predictions)[0]
        for idx in mismatched_indices:
            print(f"Index {idx}: Golden value = {Y[idx]}, Prediction = {predictions[idx]}")

    if args.benchmark:
        for version in ['tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized']:
            print(f"Benchmarking nki_predict using matmul version: {version}")
            benchmark_nki(nki_predict, X, W1, b1, W2, b2, matmul_kernel=version)
            print()

