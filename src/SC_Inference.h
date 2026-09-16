/**
 * SC_Inference.h
 * ESP32 Stochastic Computing Inference Library
 *
 * Provides portable stochastic computing primitives for running hybrid
 * stochastic-binary neural network inference on the ESP32 microcontroller.
 *
 * Based on research: "Investigating the Viability of Stochastic Computing
 * for Energy-Efficient Neural Network Inference on ESP32 Microcontrollers"
 * Abdul Ahad, TP071626, Asia Pacific University, 2026
 *
 * Key finding: SC achieves near-binary accuracy on ESP32 (97.4-98.0% vs
 * 97.6% binary on MNIST), but energy and latency are 16-20x higher than
 * binary due to the absence of dedicated bitstream-generation hardware.
 * See README for full discussion.
 *
 * MIT License — see LICENSE
 */

#ifndef SC_INFERENCE_H
#define SC_INFERENCE_H

#include <Arduino.h>
#include <stdint.h>

/**
 * Generate one 32-bit stochastic word representing probability p/255.
 *
 * Uses esp_random() — the ESP32's hardware RNG seeded from RF noise.
 * Each bit is 1 with probability (value/255), so the bit-count / 32
 * approximates the original value.
 *
 * NOTE: This is the primary performance bottleneck. Generating 64 random
 * bytes per multiplication dominates inference time, not the AND operation
 * itself. See the research paper for the full RNG characterisation results.
 *
 * @param value  Value to encode, 0-255
 * @return       32-bit stochastic word
 */
inline uint32_t sc_word(uint8_t value) {
    uint32_t r = 0;
    for (int b = 0; b < 32; b++) {
        if ((esp_random() & 0xFF) < value) r |= (1u << b);
    }
    return r;
}

/**
 * Stochastic multiply-accumulate: compute the dot product of an input
 * activation vector and a row of quantised weights using SC arithmetic.
 *
 * Each multiplication is performed by ANDing two stochastic words and
 * counting the resulting 1-bits (hardware popcount). This approximates
 * the product of the two input probabilities.
 *
 * @param inputs       Input activation array (float, post-ReLU)
 * @param weights      Quantised int8 weight row (PROGMEM)
 * @param n            Number of elements (dot product length)
 * @param weight_scale Dequantisation scale factor for this weight row
 * @return             Accumulated dot product (float)
 */
float sc_mac(const float* inputs, const int8_t* weights, int n,
             float weight_scale);

/**
 * Binary fully-connected layer with ReLU activation.
 *
 * Uses standard int8 arithmetic with per-layer dequantisation.
 * This is the fast path — use for the first and last layers of your
 * network, where approximation error would compound most.
 *
 * @param inputs        Input array (float)
 * @param weights       Quantised weight matrix, row-major (PROGMEM)
 * @param biases        Quantised bias array (PROGMEM)
 * @param outputs       Output array (float, written by this function)
 * @param n_in          Number of inputs
 * @param n_out         Number of outputs (neurons in this layer)
 * @param weight_scale  Per-layer weight dequantisation scale
 * @param bias_scale    Per-layer bias dequantisation scale
 * @param relu          Apply ReLU if true (set false for output layer)
 */
void sc_binary_layer(const float*   inputs,
                     const int8_t*  weights,
                     const int8_t*  biases,
                     float*         outputs,
                     int            n_in,
                     int            n_out,
                     float          weight_scale,
                     float          bias_scale,
                     bool           relu);

/**
 * Stochastic fully-connected layer with ReLU activation.
 *
 * The middle layer — replace sc_binary_layer with this on the hidden
 * layer(s) you want to run stochastically. Based on the layer-sensitivity
 * findings of Wang et al. (2025), keep the first and last layers binary.
 *
 * @param inputs        Input array (float, post-ReLU from previous layer)
 * @param weights       Quantised weight matrix, row-major (PROGMEM)
 * @param biases        Quantised bias array (PROGMEM)
 * @param outputs       Output array (float, written by this function)
 * @param n_in          Number of inputs
 * @param n_out         Number of outputs
 * @param weight_scale  Per-layer weight dequantisation scale
 * @param bias_scale    Per-layer bias dequantisation scale
 */
void sc_stochastic_layer(const float*  inputs,
                         const int8_t* weights,
                         const int8_t* biases,
                         float*        outputs,
                         int           n_in,
                         int           n_out,
                         float         weight_scale,
                         float         bias_scale);

/**
 * Argmax over a float array — returns index of the largest value.
 * Use this as the final step of any classification network.
 *
 * @param scores  Array of class scores (float)
 * @param n       Number of classes
 * @return        Index of the highest score (predicted class)
 */
int sc_argmax(const float* scores, int n);

#endif // SC_INFERENCE_H
