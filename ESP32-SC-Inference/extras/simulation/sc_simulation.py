
import serial
import serial.tools.list_ports
import numpy as np
import time
from torchvision import datasets, transforms
import torch

# ── Find ESP32 port ────────────────────────────────────────────────────
def find_esp32_port():
    ports = list(serial.tools.list_ports.comports())
    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} — {p.description}")

    # Try to auto-suggest the most likely ESP32 port
    suggestion = None
    for p in ports:
        desc = p.description.upper()
        if 'CP210' in desc or 'CH340' in desc or 'SILICON LABS' in desc:
            suggestion = p.device
            break

    if suggestion:
        choice = input(f"Press Enter to use suggested port {suggestion}, "
                        f"or type a port name manually: ").strip()
        return choice if choice else suggestion
    else:
        return input("Enter port manually (e.g. COM5): ").strip()

# ── Load MNIST ─────────────────────────────────────────────────────────
print("Loading MNIST test data...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

# ── Connect to ESP32 ───────────────────────────────────────────────────
port = find_esp32_port()
print(f"Connecting to ESP32 on {port}...")
ser = serial.Serial(port, 115200, timeout=5)
time.sleep(2)
ser.flushInput()
print("Connected.")

# ── Send digits and collect predictions ───────────────────────────────
n_test = 100   # increased from 20 for a statistically stronger sample
correct = 0
results = []

print(f"\nTesting {n_test} real MNIST digits...")
print(f"{'#':>4} {'True':>6} {'Pred':>6} {'Match':>6}")
print("-" * 28)

for idx in range(n_test):
    img_tensor, label = test_data[idx]

    # Convert to uint8 [0,255] — ESP32 will divide by 255 internally
    pixels = (img_tensor.numpy() * 255).astype(np.uint8)

    # Send command to ESP32 to receive raw pixels
    ser.write(b'R')           # 'R' = receive raw pixels mode
    time.sleep(0.05)

    # Send all 784 bytes
    ser.write(pixels.tobytes())
    time.sleep(0.1)

    # Read prediction back (skip any Latency/Energy/RAM lines)
    pred = None
    t0 = time.time()
    while time.time() - t0 < 5:
        response = ser.readline().decode('utf-8', errors='ignore').strip()
        if response.startswith("PRED:"):
            try:
                pred = int(response.split(':')[-1].strip())
            except ValueError:
                pred = None
            break
        if response.startswith("ERROR"):
            print(f"{idx+1:>4}  ESP32 error: {response}")
            break

    if pred is not None:
        match = pred == label
        if match:
            correct += 1
        results.append({'true': label, 'pred': pred, 'match': match})
        print(f"{idx+1:>4} {label:>6} {pred:>6} {'✓' if match else '✗':>6}")
    else:
        print(f"{idx+1:>4} {label:>6} {'ERR':>6} {'✗':>6}")
        results.append({'true': label, 'pred': -1, 'match': False})

ser.close()

# ── Summary ────────────────────────────────────────────────────────────
acc = correct / n_test * 100
print(f"\nResults: {correct}/{n_test} correct = {acc:.1f}% accuracy")
print("This is REAL inference on REAL MNIST test images on physical ESP32 hardware.")