/**
 * MNIST_Digit.ino
 * Example: Hybrid SC-Binary MLP for MNIST digit classification
 *
 * This sketch demonstrates the SC_Inference library with the 784-128-64-10
 * hybrid architecture used in the FYP research project. Layer 1 and Layer 3
 * use binary arithmetic; Layer 2 uses stochastic computing.
 *
 * Architecture:
 *   Input (784)  -> [Binary] Layer 1 (128) -> [SC] Layer 2 (64)
 *                -> [Binary] Layer 3 (10)  -> argmax -> predicted digit
 *
 * Protocol (USB serial, 115200 baud):
 *   Send 'R' followed by 784 raw pixel bytes (0-255, row-major 28x28).
 *   Board replies:
 *     PRED:<digit>
 *     Latency: <us> us
 *     Energy : <mJ> mJ
 *     RAM free: <bytes> bytes
 *
 * Or send a digit character '0'-'9' for a quick synthetic pattern test.
 *
 * Requirements:
 *   - ESP32 board (tested on ESP32 DevKitC)
 *   - Arduino IDE with ESP32 support installed
 *   - SC_Inference library installed (this library)
 *
 * See the Python scripts in extras/ for sending real MNIST images.
 */

#include <SC_Inference.h>
#include "weights.h"

// ---------------------------------------------------------------------------
// Layer dimensions  (must match weights.h)
// ---------------------------------------------------------------------------
#define L1_IN   784
#define L1_OUT  128
#define L2_IN   128
#define L2_OUT  64
#define L3_IN   64
#define L3_OUT  10

// ---------------------------------------------------------------------------
// Activation buffers (allocated once, reused each inference)
// ---------------------------------------------------------------------------
static float a1[L1_OUT];   // After Layer 1
static float a2[L2_OUT];   // After Layer 2
static float a3[L3_OUT];   // After Layer 3 (class scores)

// ---------------------------------------------------------------------------
// Full hybrid inference: raw pixels -> predicted digit class
// ---------------------------------------------------------------------------
int sc_infer(uint8_t* raw_pixels) {
    // Normalise pixel bytes to [0, 1]
    float px[L1_IN];
    for (int i = 0; i < L1_IN; i++)
        px[i] = (float)raw_pixels[i] / 255.0f;

    // Layer 1 — Binary, ReLU
    sc_binary_layer(px, fc1_weight, fc1_bias,
                    a1, L1_IN, L1_OUT,
                    SCALE_FC1_WEIGHT, SCALE_FC1_BIAS,
                    true);

    // Layer 2 — Stochastic computing, ReLU applied inside
    sc_stochastic_layer(a1, fc2_weight, fc2_bias,
                        a2, L2_IN, L2_OUT,
                        SCALE_FC2_WEIGHT, SCALE_FC2_BIAS);

    // Layer 3 — Binary, no ReLU (output scores)
    sc_binary_layer(a2, fc3_weight, fc3_bias,
                    a3, L3_IN, L3_OUT,
                    SCALE_FC3_WEIGHT, SCALE_FC3_BIAS,
                    false);

    return sc_argmax(a3, L3_OUT);
}

// ---------------------------------------------------------------------------
// Print timing + energy + RAM measurements over serial
// ---------------------------------------------------------------------------
void print_metrics(uint32_t elapsed_us) {
    float energy_mJ = (elapsed_us / 1000000.0f) * 3.3f * 0.080f * 1000.0f;
    Serial.print("Latency: "); Serial.print(elapsed_us);  Serial.println(" us");
    Serial.print("Energy : "); Serial.print(energy_mJ, 4); Serial.println(" mJ");
    Serial.print("RAM free: "); Serial.print(ESP.getFreeHeap()); Serial.println(" bytes");
}

// ---------------------------------------------------------------------------
// Binary baseline benchmark (runs once at startup)
// ---------------------------------------------------------------------------
void run_binary_baseline() {
    uint32_t t0 = micros();
    volatile int32_t acc = 0;
    for (int j = 0; j < L1_OUT; j++)
        for (int i = 0; i < L1_IN; i++)
            acc += (int32_t)100 * (int32_t)120;
    uint32_t elapsed = micros() - t0;

    Serial.println("---- BINARY BASELINE ----");
    print_metrics(elapsed);
    Serial.println("-------------------------");
}

// ---------------------------------------------------------------------------
// Synthetic test pattern (for quick offline testing without Python)
// ---------------------------------------------------------------------------
void make_synthetic_pattern(uint8_t* pixels, int digit) {
    memset(pixels, 0, 784);
    switch (digit) {
        case 0:
            for (int r = 5;  r < 23; r++) { pixels[r*28+8] = 255; pixels[r*28+19] = 255; }
            for (int c = 8;  c < 20; c++) { pixels[5*28+c] = 255; pixels[22*28+c] = 255; }
            break;
        case 1:
            for (int r = 4; r < 24; r++) { pixels[r*28+13] = 255; pixels[r*28+14] = 255; }
            break;
        case 7:
            for (int c = 6; c < 20; c++) pixels[4*28+c] = 255;
            for (int r = 4; r < 24; r++) pixels[r*28+19] = 255;
            break;
        default:
            // No reliable pattern for other digits — fill with a grey square
            for (int r = 6; r < 22; r++)
                for (int c = 6; c < 22; c++)
                    pixels[r*28+c] = 128;
            Serial.println("WARNING: no unique synthetic pattern for this digit.");
            Serial.println("Send real MNIST images via Python for reliable accuracy.");
            break;
    }
}

// ---------------------------------------------------------------------------
// setup
// ---------------------------------------------------------------------------
void setup() {
    Serial.setRxBufferSize(2048);
    Serial.begin(115200);
    delay(3000);

    Serial.println("==================================");
    Serial.println("ESP32 SC Inference Library — MNIST");
    Serial.println("Hybrid Binary-SC-Binary MLP");
    Serial.println("==================================");

    run_binary_baseline();

    Serial.println("Ready.");
    Serial.println("  Send 'R' + 784 bytes  -> real MNIST inference");
    Serial.println("  Send digit '0'-'9'    -> synthetic pattern test");
}

// ---------------------------------------------------------------------------
// loop
// ---------------------------------------------------------------------------
void loop() {
    if (!Serial.available()) return;

    char cmd = Serial.read();

    // ── Real MNIST image from Python ──────────────────────────────────
    if (cmd == 'R') {
        uint8_t pixels[784];
        int received = 0;
        unsigned long t0 = millis();
        while (received < 784 && millis() - t0 < 3000) {
            if (Serial.available()) pixels[received++] = Serial.read();
        }
        if (received != 784) {
            Serial.print("ERROR: received "); Serial.print(received);
            Serial.println("/784 bytes. Timeout?");
            return;
        }
        uint32_t sc_start = micros();
        int pred = sc_infer(pixels);
        uint32_t sc_elapsed = micros() - sc_start;
        Serial.print("PRED:"); Serial.println(pred);
        print_metrics(sc_elapsed);
    }

    // ── Synthetic digit pattern test ──────────────────────────────────
    else if (cmd >= '0' && cmd <= '9') {
        int digit = cmd - '0';
        uint8_t pixels[784];
        make_synthetic_pattern(pixels, digit);

        uint32_t t0 = micros();
        int pred = sc_infer(pixels);
        uint32_t elapsed = micros() - t0;

        Serial.print("Input   : "); Serial.println(digit);
        Serial.print("PRED    : "); Serial.println(pred);
        print_metrics(elapsed);
        if (digit != 0 && digit != 1 && digit != 7)
            Serial.println("(Note: result unreliable — use Python sender for real accuracy)");
    }
}
