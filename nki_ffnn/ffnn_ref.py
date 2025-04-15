import os
import time
import argparse
import numpy as np

from utils import BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, generate_data, load_data, save_data, save_results

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability fix
    return e_x / np.sum(e_x, axis=1, keepdims=True)

class NeuralNetwork:
    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    def forward(self, X):
        # Layer 1
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        
        # Layer 2 (output)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = softmax(self.z2)

        return self.a2

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1).astype(np.int32)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feedforward Neural Network")
    parser.add_argument("--seed", type=int, default=152, help="Random seed for reproducibility")
    parser.add_argument("--load-data", action="store_true", help="Whether to load input, weights, and biases from files or generate them")
    parser.add_argument("--store-data", action="store_true", help="Store input, weights, biases, and results to file")
    parser.add_argument("--benchmark", action="store_true", help="Whether to run the benchmarking of the forward pass")
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.load_data:
        print("Loading input, weights, and biases...")
        X, W1, b1, W2, b2 = load_data()
    else:
        print("Generating random input, weights, and biases...")
        X, W1, b1, W2, b2 = generate_data()

    nn = NeuralNetwork(W1, b1, W2, b2)
    
    predictions = nn.predict(X)
    print("Predicted class indices:")
    print(predictions)

    if args.store_data:
        print("Storing input, weights, biases, and results...")
        save_data(X, W1, b1, W2, b2)
        save_results(predictions)

    if args.benchmark:
        print("Benchmarking prediction on ffnn...")
        num_iterations = 10
        start_time = time.time()
        for _ in range(num_iterations):
            nn.predict(X)
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        print(f"Average time for prediction on ffnn: {avg_time*1e3:.6f} ms")

