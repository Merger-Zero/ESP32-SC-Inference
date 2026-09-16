# ESP32-SC-Inference

**Stochastic computing inference library for ESP32 microcontrollers**

Portable C++ primitives for running hybrid stochastic-binary neural networks on the ESP32 using Arduino IDE. Based on the Final Year Project research at Asia Pacific University of Technology and Innovation, 2026.

---

## What is stochastic computing?

Stochastic computing (SC) represents numerical values as probabilities encoded in random bitstreams, and performs multiplication with a single AND gate instead of a conventional multiplier circuit. Published research on FPGAs and custom ASICs reports large energy savings from this approach.

This library lets you test that approach on a commodity ESP32 — no FPGA toolchain or chip design required.

---

## Key research finding

This library was built as part of an empirical study comparing SC on commodity hardware against FPGA/ASIC literature claims. The result was a clear split:

| Metric | SC vs Binary on ESP32 |
|---|---|
| MNIST classification accuracy | 97.4-98.0% vs 97.6% binary (within 0.3%) |
| Latency (stream=32 bits) | 74,273 µs vs 3,782 µs (19.6x more) |
| Energy (stream=32 bits) | 19.6 mJ vs 1.2194 mJ (16.1x more) |

**Accuracy transfers to commodity hardware. Energy savings do not** — because the ESP32 has no dedicated parallel bitstream-generation hardware. The bottleneck is generating 64 random values per multiplication in sequential software, not the AND operation itself.

This contrasts with Lee et al. (2024) reporting 99.72% energy savings on FPGA, and Wang et al. (2025) reporting 49.8% savings on a 32nm ASIC. Those savings are real — but they depend on dedicated hardware this library cannot replicate on a general-purpose microcontroller.

See the full research paper for the complete analysis.

---

## Installation

### Via Arduino IDE (recommended)

1. Download this repository as a ZIP
2. In Arduino IDE: **Sketch > Include Library > Add .ZIP Library**
3. Select the downloaded ZIP

### Manual

Copy the `ESP32-SC-Inference` folder into your Arduino `libraries/` directory.

---

## Quick start

```cpp
#include <SC_Inference.h>
#include "weights.h"   // your quantised model weights

float a1[128], a2[64], a3[10];

int classify(uint8_t* pixels) {
    float px[784];
    for (int i = 0; i < 784; i++) px[i] = pixels[i] / 255.0f;

    // Layer 1: binary (fast, accurate)
    sc_binary_layer(px, fc1_weight, fc1_bias,
                    a1, 784, 128,
                    SCALE_FC1_WEIGHT, SCALE_FC1_BIAS, true);

    // Layer 2: stochastic (the SC layer)
    sc_stochastic_layer(a1, fc2_weight, fc2_bias,
                        a2, 128, 64,
                        SCALE_FC2_WEIGHT, SCALE_FC2_BIAS);

    // Layer 3: binary output
    sc_binary_layer(a2, fc3_weight, fc3_bias,
                    a3, 64, 10,
                    SCALE_FC3_WEIGHT, SCALE_FC3_BIAS, false);

    return sc_argmax(a3, 10);
}
```

---

## API reference

### `sc_word(uint8_t value)`
Generates one 32-bit stochastic word using `esp_random()` (the ESP32's hardware RNG). Each bit is 1 with probability `value/255`. Returns a `uint32_t`.

### `sc_binary_layer(...)`
Binary fully-connected layer with optional ReLU. Use this for the first and last layers of your network — early layers are most sensitive to approximation error.

### `sc_stochastic_layer(...)`
Stochastic computing fully-connected layer with ReLU. Replace `sc_binary_layer` with this on one middle hidden layer for the hybrid SC-Binary architecture.

### `sc_argmax(float* scores, int n)`
Returns the index of the highest value in an array. Use as the final step of a classification network.

---

## Running the MNIST example

1. Open `examples/MNIST_Digit/MNIST_Digit.ino` in Arduino IDE
2. Flash to your ESP32
3. Open Serial Monitor at 115200 baud
4. Send digit character `0`, `1`, or `7` for a quick synthetic pattern test
5. Or use `extras/simulation/sc_simulation.py` to send real MNIST test images:

```bash
pip install pyserial torch torchvision
python extras/simulation/sc_simulation.py --port COM3 --n_test 100
```

---

## Generating your own weights

To use this library with your own model:

1. Train a PyTorch MLP and quantise weights to int8
2. Export each layer's weights, biases, and per-layer scale factors
3. Format as `const int8_t array[] PROGMEM` arrays in a `weights.h` file
4. Define `SCALE_<LAYER>_WEIGHT` and `SCALE_<LAYER>_BIAS` constants

See `extras/training/float_baseline.py` for a complete training and export example, and `examples/MNIST_Digit/weights.h` for the expected format.

**Important:** use the actual per-layer scale factor for dequantisation, not a fixed constant like 127. Using a fixed divisor is the single biggest source of accuracy loss in SC implementations on microcontrollers, and was the main bug found and fixed in the research that produced this library.

---

## File structure

```
ESP32-SC-Inference/
├── src/
│   ├── SC_Inference.h       <- portable SC primitives (include this)
│   └── SC_Inference.cpp     <- implementation
├── examples/
│   └── MNIST_Digit/
│       ├── MNIST_Digit.ino  <- full 784-128-64-10 hybrid MLP example
│       └── weights.h        <- pre-trained MNIST weights (PROGMEM)
├── extras/
│   ├── simulation/
│   │   └── sc_simulation.py <- send real MNIST images to ESP32 over serial
│   └── training/
│       ├── float_baseline.py        <- float32 accuracy reference
│       └── architecture_comparison.py  <- compare SC overhead across architectures
├── library.properties
├── LICENSE
└── README.md
```

---

## References

- Lee, Y. Y. et al. (2024). FPGA-optimised stochastic CNN. Research, 7, 0307.
- Wang, Z. et al. (2025). Energy-efficient SC networks with layer-wise ASL. IEEE IoT Journal, 12(14).
- Liu, Y. et al. (2021). A survey of stochastic computing neural networks. IEEE TNNLS, 32(7).

---

## Citation

If you use this library in your work, please cite the original research:

> Abdul Ahad (2026). *Investigating the Viability of Stochastic Computing for Energy-Efficient Neural Network Inference on ESP32 Microcontrollers*. Final Year Project, Asia Pacific University of Technology and Innovation.

---

## License

MIT License. See `LICENSE` for full text.
