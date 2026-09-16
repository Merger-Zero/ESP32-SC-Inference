
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gzip
import urllib.request
import os

SAMPLE_COUNT = 5       # kept small — full-network SC sim is expensive
STREAM_LENGTH = 32
EPOCHS = 3

# ── MNIST loader (same as before) ───────────────────────────────────
def load_mnist(n_train=6000, n_test=1000, data_dir="./data_raw"):
    os.makedirs(data_dir, exist_ok=True)
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    paths = {}
    for key, fname in files.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            urllib.request.urlretrieve(base_url + fname, fpath)
        paths[key] = fpath

    def read_images(path, n):
        with gzip.open(path, "rb") as f:
            f.read(16)
            buf = f.read(n * 28 * 28)
            return np.frombuffer(buf, dtype=np.uint8).reshape(n, 1, 28, 28)

    def read_labels(path, n):
        with gzip.open(path, "rb") as f:
            f.read(8)
            buf = f.read(n)
            return np.frombuffer(buf, dtype=np.uint8)

    x_train = read_images(paths["train_images"], n_train).astype(np.float32) / 255.0
    y_train = read_labels(paths["train_labels"], n_train)
    x_test  = read_images(paths["test_images"], n_test).astype(np.float32) / 255.0
    y_test  = read_labels(paths["test_labels"], n_test)
    return x_train, y_train, x_test, y_test


print("Loading MNIST...")
x_train, y_train, x_test, y_test = load_mnist()
x_train_t = torch.tensor(x_train)
y_train_t = torch.tensor(y_train, dtype=torch.long)
x_test_t  = torch.tensor(x_test)
y_test_t  = torch.tensor(y_test, dtype=torch.long)


# ── Same 5 architectures as before ──────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc1(x)


class MobileNetStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = nn.Conv2d(1, 1, 3, padding=1, groups=1)
        self.pointwise = nn.Conv2d(1, 8, 1)
        self.depthwise2 = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.pointwise2 = nn.Conv2d(8, 16, 1)
        self.fc1 = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.pointwise(self.depthwise(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.pointwise2(self.depthwise2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc1(x)


class FireModule(nn.Module):
    def __init__(self, in_ch, squeeze_ch, expand_ch):
        super().__init__()
        self.squeeze = nn.Conv2d(in_ch, squeeze_ch, 1)
        self.expand1x1 = nn.Conv2d(squeeze_ch, expand_ch, 1)
        self.expand3x3 = nn.Conv2d(squeeze_ch, expand_ch, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.squeeze(x))
        return torch.cat([F.relu(self.expand1x1(x)), F.relu(self.expand3x3(x))], dim=1)


class SqueezeNetStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.fire1 = FireModule(8, 4, 8)
        self.fc1 = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.fire1(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + x)


class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.res1 = ResidualBlock(8)
        self.res2 = ResidualBlock(8)
        self.fc1 = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = F.max_pool2d(x, 2)
        x = self.res2(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)


# ── Training (unchanged) ────────────────────────────────────────────
def train_model(model, name):
    print(f"\nTraining {name}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    batch_size = 64
    n = x_train_t.shape[0]

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        correct, total = 0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = x_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
        print(f"  Epoch {epoch+1}/{EPOCHS} — train accuracy: {correct/total*100:.2f}%")

    model.eval()
    with torch.no_grad():
        preds = model(x_test_t).argmax(1)
        acc = (preds == y_test_t).float().mean().item() * 100
    print(f"  {name} binary test accuracy: {acc:.2f}%")
    return acc


# ── SC primitives (same as rest of project) ─────────────────────────
def sc_matmul(a, w, stream_length):
    batch, in_f = a.shape
    out_f = w.shape[0]
    a_abs = np.abs(a)
    w_abs = np.abs(w)
    result = np.zeros((batch, out_f), dtype=np.float32)
    for b in range(batch):
        x_streams = (np.random.random((out_f, in_f, stream_length))
                     < a_abs[b][None, :, None]).astype(np.uint8)
        w_streams = (np.random.random((out_f, in_f, stream_length))
                     < w_abs[:, :, None]).astype(np.uint8)
        products = np.mean(np.bitwise_and(x_streams, w_streams), axis=2)
        signs = np.sign(w) * np.sign(a[b])[None, :]
        result[b] = np.sum(signs * products, axis=1)
    return result


def im2col(x, kernel_size, padding, stride=1):
    x_t = torch.tensor(x)
    unfolded = F.unfold(x_t, kernel_size=kernel_size, padding=padding, stride=stride)
    return unfolded.numpy()


def sc_conv_layer(x_np, weight_np, kernel_size, padding, stream_length, groups=1):
    batch = x_np.shape[0]
    out_ch = weight_np.shape[0]

    if groups == 1:
        cols = im2col(x_np, kernel_size, padding)
        w_flat = weight_np.reshape(out_ch, -1)
    else:
        # Depthwise: each input channel only sees its own filter.
        # Process per-group to keep this correct and simple.
        in_ch_per_group = x_np.shape[1] // groups
        out_ch_per_group = out_ch // groups
        outs = []
        for g in range(groups):
            xg = x_np[:, g*in_ch_per_group:(g+1)*in_ch_per_group]
            wg = weight_np[g*out_ch_per_group:(g+1)*out_ch_per_group]
            cols_g = im2col(xg, kernel_size, padding)
            w_flat_g = wg.reshape(out_ch_per_group, -1)
            max_a = np.max(np.abs(cols_g)) + 1e-8
            max_w = np.max(np.abs(w_flat_g)) + 1e-8
            n_pos = cols_g.shape[2]
            out_g = np.zeros((batch, out_ch_per_group, n_pos), dtype=np.float32)
            for p in range(n_pos):
                out_g[:, :, p] = sc_matmul(cols_g[:, :, p]/max_a, w_flat_g/max_w, stream_length)
            outs.append(out_g)
        return np.concatenate(outs, axis=1)

    max_a = np.max(np.abs(cols)) + 1e-8
    max_w = np.max(np.abs(w_flat)) + 1e-8
    cols_norm = cols / max_a
    w_norm = w_flat / max_w
    n_positions = cols.shape[2]
    out = np.zeros((batch, out_ch, n_positions), dtype=np.float32)
    for p in range(n_positions):
        out[:, :, p] = sc_matmul(cols_norm[:, :, p], w_norm, stream_length)
    return out


# ── Full-network SC timing via forward hooks ────────────────────────
def run_full_network_sc(model, name, x_sample):
    """
    Registers a hook on every Conv2d and Linear layer to capture its REAL
    input during an actual forward pass, then runs SC simulation on every
    one of those real inputs, summing total time across the whole network.
    """
    captured = []

    def hook_fn(module, input, output):
        captured.append((module, input[0].detach().numpy()))

    handles = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(torch.tensor(x_sample))

    for h in handles:
        h.remove()

    total_time = 0.0
    layer_count = 0
    for module, real_input in captured:
        layer_count += 1
        if isinstance(module, nn.Linear):
            w = module.weight.detach().numpy()
            x_flat = real_input.reshape(real_input.shape[0], -1)
            max_a = np.max(np.abs(x_flat)) + 1e-8
            max_w = np.max(np.abs(w)) + 1e-8
            t0 = time.time()
            _ = sc_matmul(x_flat / max_a, w / max_w, STREAM_LENGTH)
            total_time += time.time() - t0

        elif isinstance(module, nn.Conv2d):
            w = module.weight.detach().numpy()
            k = w.shape[-1]
            pad = module.padding[0]
            groups = module.groups
            t0 = time.time()
            _ = sc_conv_layer(real_input, w, k, pad, STREAM_LENGTH, groups=groups)
            total_time += time.time() - t0

    per_sample_us = (total_time / x_sample.shape[0]) * 1_000_000
    print(f"  {name}: {layer_count} layers simulated, "
          f"total SC time for {x_sample.shape[0]} samples = {total_time:.3f}s "
          f"({per_sample_us:.0f} us/sample)")
    return total_time, per_sample_us, layer_count


# ── Run everything ────────────────────────────────────────────────────
results = []
models = [
    (MLP(), "MLP"),
    (SmallCNN(), "Small CNN (LeNet-style)"),
    (MobileNetStyle(), "MobileNet-style"),
    (SqueezeNetStyle(), "SqueezeNet-style"),
    (SmallResNet(), "Small ResNet"),
]

x_sample = x_test_t[:SAMPLE_COUNT].numpy()

for model, name in models:
    acc = train_model(model, name)
    total_time, per_sample_us, layer_count = run_full_network_sc(model, name, x_sample)
    n_params = sum(p.numel() for p in model.parameters())
    results.append({
        "name": name, "accuracy": acc, "params": n_params,
        "layers": layer_count, "sc_time_per_sample_us": per_sample_us,
    })

print("\n" + "=" * 90)
print("FULL-NETWORK COMPARISON — EVERY LAYER SUMMED, ALL VALUES MEASURED")
print("=" * 90)
print(f"{'Model':<28} {'Params':>10} {'Layers':>7} {'Accuracy':>10} {'SC us/sample':>15} {'Overhead vs MLP':>18}")
print("-" * 90)
baseline = results[0]["sc_time_per_sample_us"]
for r in results:
    overhead = r["sc_time_per_sample_us"] / baseline
    print(f"{r['name']:<28} {r['params']:>10,} {r['layers']:>7} {r['accuracy']:>9.2f}% "
          f"{r['sc_time_per_sample_us']:>14.0f}u {overhead:>16.2f}x")

print("\nDone. Every number above sums real SC simulation across every layer.")