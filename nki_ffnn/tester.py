import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt

import argparse
import numpy as np
from kernels import nki_transpose, nki_bias_add_act, nki_forward, nki_predict
from ffnn_ref import NeuralNetwork, relu, softmax
from utils import generate_data

def test_transpose(simulate):
    print("Testing transpose...")
    if simulate:
        for test_i in range(10):
            input_matrix = np.random.rand(1024, 1024).astype(np.float32)
            expected_output = input_matrix.T
            output = nki.simulate_kernel(nki_transpose, input_matrix)
            assert np.allclose(output, expected_output), "Transpose test failed!"
    else:
        input_matrix = np.random.rand(1024, 1024).astype(np.float32)
        expected_output = input_matrix.T
        output = nki_transpose(input_matrix)
        assert np.allclose(output, expected_output), "Transpose test failed!"
    print("Transpose test passed!")

def test_bias_add_act(simulate):
    print("Testing bias_add_act...")
    if simulate:
        for test_i in range(10):
            input_matrix = np.random.rand(1024, 1024).astype(np.float32)
            bias = np.random.rand(1, 1024).astype(np.float32)
            expected_output = relu(input_matrix + bias)
            output = nki.simulate_kernel(nki_bias_add_act, input_matrix, bias, act='relu')
            assert np.allclose(output, expected_output), "Bias add + activation(relu) test failed!"
        for test_i in range(10):
            input_matrix = np.random.rand(1024, 1024).astype(np.float32)
            bias = np.random.rand(1, 1024).astype(np.float32)
            expected_output = softmax(input_matrix + bias)
            output = nki.simulate_kernel(nki_bias_add_act, input_matrix, bias, act='softmax')
            assert np.allclose(output, expected_output), "Bias add + activation(softmax) test failed!"
    else:
        input_matrix = np.random.rand(1024, 1024).astype(np.float32)
        bias = np.random.rand(1, 1024).astype(np.float32)
        expected_output = relu(input_matrix + bias)
        output = nki_bias_add_act(input_matrix, bias, act='relu')
        assert np.allclose(output, expected_output), "Bias add + activation(relu) test failed!"
        input_matrix = np.random.rand(1024, 1024).astype(np.float32)
        bias = np.random.rand(1, 1024).astype(np.float32)
        expected_output = softmax(input_matrix + bias)
        output = nki_bias_add_act(input_matrix, bias, act='softmax')
        assert np.allclose(output, expected_output), "Bias add + activation(softmax) test failed!"
    print("Bias add + activation test passed!")

def test_forward(simulate):
    print("Testing forward...")
    if simulate:
        for test_i in range(3):
            X = np.random.rand(1024, 1024).astype(np.float32)
            W1 = (np.random.rand(1024, 1024) * 0.01).astype(np.float32)
            W2 = (np.random.rand(1024, 1024) * 0.01).astype(np.float32)
            b1 = (np.random.rand(1, 1024) * 0.01).astype(np.float32)
            b2 = (np.random.rand(1, 1024) * 0.01).astype(np.float32)
            nn = NeuralNetwork(W1, b1, W2, b2)
            expected_output = nn.forward(X)
            output = nki.simulate_kernel(nki_forward, X, W1, b1, W2, b2)
            assert np.allclose(output, expected_output), f"Forward test failed!\nNKI Output:\n{output}\nExpected Output:\n{expected_output}"
    else:
        X = np.random.rand(1024, 1024).astype(np.float32)
        W1 = (np.random.rand(1024, 1024) * 0.01).astype(np.float32)
        W2 = (np.random.rand(1024, 1024) * 0.01).astype(np.float32)
        b1 = (np.random.rand(1, 1024) * 0.01).astype(np.float32)
        b2 = (np.random.rand(1, 1024) * 0.01).astype(np.float32)
        nn = NeuralNetwork(W1, b1, W2, b2)
        expected_output = nn.forward(X)
        output = nki_forward(X, W1, b1, W2, b2)
        assert np.allclose(output, expected_output), f"Forward test failed!\nNKI Output:\n{output}\nExpected Output:\n{expected_output}"
    print("Forward test passed!")

def test_predict(simulate):
    print("Testing predict...")
    if simulate:
        for test_i in range(3):
            X = np.random.rand(1024, 1024).astype(np.float32)
            W1 = (np.random.rand(1024, 1024) * 0.01).astype(np.float32)
            W2 = (np.random.rand(1024, 1024) * 0.01).astype(np.float32)
            b1 = (np.random.rand(1, 1024) * 0.01).astype(np.float32)
            b2 = (np.random.rand(1, 1024) * 0.01).astype(np.float32)
            nn = NeuralNetwork(W1, b1, W2, b2)
            expected_output = nn.predict(X)
            output = nki.simulate_kernel(nki_predict, X, W1, b1, W2, b2)
            assert np.allclose(output, expected_output), "Predict test failed!"
    else:
        X = np.random.rand(1024, 1024).astype(np.float32)
        W1 = (np.random.rand(1024, 1024) * 0.01).astype(np.float32)
        W2 = (np.random.rand(1024, 1024) * 0.01).astype(np.float32)
        b1 = (np.random.rand(1, 1024) * 0.01).astype(np.float32)
        b2 = (np.random.rand(1, 1024) * 0.01).astype(np.float32)
        nn = NeuralNetwork(W1, b1, W2, b2)
        expected_output = nn.predict(X)
        output = nki_predict(X, W1, b1, W2, b2)
        assert np.allclose(output, expected_output), "Predict test failed!"
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