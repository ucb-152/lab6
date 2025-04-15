import os
import numpy as np

BATCH_SIZE = 1024*2
INPUT_SIZE = 1024*4
HIDDEN_SIZE = 1024*8
OUTPUT_SIZE = 1024*2

def generate_data():
    X = np.random.rand(BATCH_SIZE, INPUT_SIZE).astype(np.float32)
    W1 = (np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.01).astype(np.float32)
    b1 = (np.random.randn(1, HIDDEN_SIZE) * 0.01).astype(np.float32)
    W2 = (np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.01).astype(np.float32)
    b2 = (np.random.randn(1, OUTPUT_SIZE) * 0.01).astype(np.float32)

    return X, W1, b1, W2, b2

def save_data(X, W1, b1, W2, b2, path="ffnn"):
    os.makedirs(path, exist_ok=True)

    # Save input sample
    with open(f"{path}/X.bin", "wb") as f:
        X.tofile(f)

    # Save pre-trained weights
    with open(f"{path}/W1.bin", "wb") as f:
        W1.tofile(f)
    with open(f"{path}/b1.bin", "wb") as f:
        b1.tofile(f)
    with open(f"{path}/W2.bin", "wb") as f:
        W2.tofile(f)
    with open(f"{path}/b2.bin", "wb") as f:
        b2.tofile(f)

def load_data(path="ffnn"):
    # Load input sample
    with open(f"{path}/X.bin", "rb") as f:
        X = np.fromfile(f, dtype=np.float32).reshape(BATCH_SIZE, INPUT_SIZE)

    # Load pre-trained weights
    with open(f"{path}/W1.bin", "rb") as f:
        W1 = np.fromfile(f, dtype=np.float32).reshape(INPUT_SIZE, HIDDEN_SIZE)
    with open(f"{path}/b1.bin", "rb") as f:
        b1 = np.fromfile(f, dtype=np.float32).reshape(1, HIDDEN_SIZE)
    with open(f"{path}/W2.bin", "rb") as f:
        W2 = np.fromfile(f, dtype=np.float32).reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    with open(f"{path}/b2.bin", "rb") as f:
        b2 = np.fromfile(f, dtype=np.float32).reshape(1, OUTPUT_SIZE)

    return X, W1, b1, W2, b2

def save_results(Y, path="ffnn"):
    os.makedirs(path, exist_ok=True)

    # Save expected output
    with open(f"{path}/Y.bin", "wb") as f:
        Y.tofile(f)

def load_results(path="ffnn"):
    # Load expected output
    with open(f"{path}/Y.bin", "rb") as f:
        Y = np.fromfile(f, dtype=np.int32)
    return Y
