
import numpy as np
import torch
import torch.nn as nn
import gzip
import urllib.request
import os

# ── Load MNIST test set ──────────────────────────────────────────────
def load_mnist_test(n=1000, data_dir="./data_raw"):
    os.makedirs(data_dir, exist_ok=True)
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "images": "t10k-images-idx3-ubyte.gz",
        "labels": "t10k-labels-idx1-ubyte.gz",
    }
    paths = {}
    for key, fname in files.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            urllib.request.urlretrieve(base_url + fname, fpath)
        paths[key] = fpath

    with gzip.open(paths["images"], "rb") as f:
        f.read(16)
        buf = f.read(n * 28 * 28)
        images = np.frombuffer(buf, dtype=np.uint8).reshape(n, 784)

    with gzip.open(paths["labels"], "rb") as f:
        f.read(8)
        buf = f.read(n)
        labels = np.frombuffer(buf, dtype=np.uint8)

    images = images.astype(np.float32) / 255.0
    return images, labels

# ── Model definition (must match training) ─────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ── Load model and test set ─────────────────────────────────────────────
print("Loading model...")
model = MLP()
model.load_state_dict(torch.load("mnist_mlp.pt", weights_only=True))
model.eval()

print("Loading test data...")
x_test, y_test = load_mnist_test(n=1000)

# ── Run inference using ORIGINAL float32 weights, no quantisation ──────
print("Running float32 (no quantisation) inference...")
with torch.no_grad():
    preds = model(torch.tensor(x_test)).argmax(1).numpy()

accuracy = (preds == y_test).mean() * 100

print("\n==================================")
print("FLOAT32 BASELINE (NO QUANTISATION)")
print("==================================")
print(f"Test samples : {len(y_test)}")
print(f"Accuracy     : {accuracy:.2f}%")
print("==================================")
print("\nThis is the theoretical upper-bound accuracy —")
print("the model's true performance before any weight")
print("compression or stochastic approximation is applied.")