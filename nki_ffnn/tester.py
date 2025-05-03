import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt

import os
import argparse
import numpy as np
from kernels import nki_transpose, nki_bias_add_act, nki_forward, nki_predict
from ffnn_ref import NeuralNetwork, relu, softmax
from utils import generate_data, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE

def write_outputs_to_file(test_result, ref_result, file_head):
    # Create output directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Write reference output to a text file
    ref_output_file = os.path.join(output_dir, f"{file_head}_ref.txt")
    with open(ref_output_file, "w") as ref_file:
        np.savetxt(ref_file, ref_result, fmt="%.4f")

    # Write test output to a text file
    test_output_file = os.path.join(output_dir, f"{file_head}.txt")
    with open(test_output_file, "w") as test_file:
        np.savetxt(test_file, test_result, fmt="%.4f")

def test_transpose(simulate):
    print("Testing transpose...")

    input_matrix = np.random.rand(BATCH_SIZE, INPUT_SIZE).astype(np.float32)
    expected_output = input_matrix.T
    if simulate:
        output = nki.simulate_kernel(nki_transpose, input_matrix)
    else:
        output = nki_transpose(input_matrix)
    if not np.allclose(output, expected_output, atol=1e-7, rtol=1e-4):
        write_outputs_to_file(output, expected_output, f"transpose_test")
        raise ValueError("Transpose test failed!")

    input_matrix = np.random.rand(BATCH_SIZE, HIDDEN_SIZE).astype(np.float32)
    expected_output = input_matrix.T
    if simulate:
        output = nki.simulate_kernel(nki_transpose, input_matrix)
    else:
        output = nki_transpose(input_matrix)
    if not np.allclose(output, expected_output, atol=1e-7, rtol=1e-4):
        write_outputs_to_file(output, expected_output, f"transpose_test")
        raise ValueError("Transpose test failed!")

    print("Transpose test passed!")

def test_bias_add_act(simulate):
    print("Testing bias_add_act...")

    input_matrix = np.random.rand(BATCH_SIZE, HIDDEN_SIZE).astype(np.float32)
    bias = np.random.rand(1, HIDDEN_SIZE).astype(np.float32)
    expected_output = relu(input_matrix + bias)
    if simulate:
        output = nki.simulate_kernel(nki_bias_add_act, input_matrix, bias, act='relu')
    else:
        output = nki_bias_add_act(input_matrix, bias, act='relu')
    if not np.allclose(output, expected_output, atol=1e-7, rtol=1e-4):
        write_outputs_to_file(output, expected_output, f"bias_add_act_relu_test")
        raise ValueError("Bias add + activation(relu) test failed!")

    input_matrix = np.random.rand(BATCH_SIZE, OUTPUT_SIZE).astype(np.float32)
    bias = np.random.rand(1, OUTPUT_SIZE).astype(np.float32)
    expected_output = softmax(input_matrix + bias)
    if simulate:
        output = nki.simulate_kernel(nki_bias_add_act, input_matrix, bias, act='softmax')
    else:
        output = nki_bias_add_act(input_matrix, bias, act='softmax')
    if not np.allclose(output, expected_output, atol=1e-7, rtol=1e-4):
        write_outputs_to_file(output, expected_output, f"bias_add_act_softmax_test")
        raise ValueError("Bias add + activation(softmax) test failed!")

    print("Bias add + activation test passed!")

def test_forward(simulate):
    print("Testing forward...")

    X, W1, b1, W2, b2 = generate_data()
    nn = NeuralNetwork(W1, b1, W2, b2)
    expected_output = nn.forward(X)
    if simulate:
        output = nki.simulate_kernel(nki_forward, X, W1, b1, W2, b2)
    else:
        output = nki_forward(X, W1, b1, W2, b2)
    if not np.allclose(expected_output, output, atol=1e-7, rtol=1e-4):
        write_outputs_to_file(output, expected_output, f"forward_test")
        raise ValueError("Forward test failed!")

    print("Forward test passed!")

def test_predict(simulate):
    print("Testing predict...")

    X, W1, b1, W2, b2 = generate_data()
    nn = NeuralNetwork(W1, b1, W2, b2)
    expected_output = nn.predict(X)
    if simulate:
        output = nki.simulate_kernel(nki_predict, X, W1, b1, W2, b2)
    else:
        output = nki_predict(X, W1, b1, W2, b2)
    if not np.allclose(output, expected_output, atol=1e-7, rtol=1e-4):
        write_outputs_to_file(output, expected_output, f"predict_test")
        raise ValueError("Predict test failed!")
    print("Predict test passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test kernels in ffnn.py")
    parser.add_argument("--test-transpose", action="store_true", help="Test the transpose kernel")
    parser.add_argument("--test-bias-add-act", action="store_true", help="Test the bias_add_act kernel")
    parser.add_argument("--test-forward", action="store_true", help="Test the forward kernel")
    parser.add_argument("--test-predict", action="store_true", help="Test the predict kernel")
    parser.add_argument("--test-all", action="store_true", help="Test all kernels")
    parser.add_argument("--simulate", action="store_true", help="Run the tests in simulation mode")

    args = parser.parse_args()
    simulate = args.simulate

    np.random.seed(152)

    if args.test_all:
        test_transpose(simulate)
        test_bias_add_act(simulate)
        test_forward(simulate)
        test_predict(simulate)
    else:
        if args.test_transpose:
            test_transpose(simulate)
        if args.test_bias_add_act:
            test_bias_add_act(simulate)
        if args.test_forward:
            test_forward(simulate)
        if args.test_predict:
            test_predict(simulate)