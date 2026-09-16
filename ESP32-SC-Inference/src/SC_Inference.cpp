/**
 * SC_Inference.cpp
 * ESP32 Stochastic Computing Inference Library — implementation
 *
 * MIT License — Abdul Ahad, TP071626, APU, 2026
 */

#include "SC_Inference.h"
#include <math.h>

// ---------------------------------------------------------------------------
// sc_mac
// ---------------------------------------------------------------------------
float sc_mac(const float* inputs, const int8_t* weights, int n,
             float weight_scale) {
    // Find normalisation ranges so values map into [0, 255]
    float max_a = 1e-8f;
    for (int i = 0; i < n; i++)
        if (inputs[i] > max_a) max_a = inputs[i];

    float max_w = 1e-8f;
    for (int i = 0; i < n; i++) {
        float wf = fabsf((float)(int8_t)pgm_read_byte(&weights[i]) / weight_scale);
        if (wf > max_w) max_w = wf;
    }

    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        uint8_t px = (uint8_t)((inputs[i] / max_a) * 255.0f);
        int8_t  raw_w = (int8_t)pgm_read_byte(&weights[i]);
        float   wf    = (float)raw_w / weight_scale;
        uint8_t pw    = (uint8_t)((fabsf(wf) / max_w) * 255.0f);

        // SC multiply: AND two stochastic words, count 1-bits
        uint32_t sr = sc_word(px) & sc_word(pw);
        float product = (float)__builtin_popcount(sr) / 32.0f;

        // Preserve weight sign
        acc += (wf >= 0.0f ? 1.0f : -1.0f) * product;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// sc_binary_layer
// ---------------------------------------------------------------------------
void sc_binary_layer(const float*   inputs,
                     const int8_t*  weights,
                     const int8_t*  biases,
                     float*         outputs,
                     int            n_in,
                     int            n_out,
                     float          weight_scale,
                     float          bias_scale,
                     bool           relu) {
    for (int j = 0; j < n_out; j++) {
        vTaskDelay(0);   // yield to RTOS watchdog
        float acc = 0.0f;
        for (int i = 0; i < n_in; i++) {
            int8_t w = (int8_t)pgm_read_byte(&weights[j * n_in + i]);
            acc += inputs[i] * ((float)w / weight_scale);
        }
        int8_t b = (int8_t)pgm_read_byte(&biases[j]);
        float val = acc + (float)b / bias_scale;
        outputs[j] = (relu && val < 0.0f) ? 0.0f : val;
    }
}

// ---------------------------------------------------------------------------
// sc_stochastic_layer
// ---------------------------------------------------------------------------
void sc_stochastic_layer(const float*  inputs,
                         const int8_t* weights,
                         const int8_t* biases,
                         float*        outputs,
                         int           n_in,
                         int           n_out,
                         float         weight_scale,
                         float         bias_scale) {
    // Pre-compute per-layer normalisation once (cheaper than per-neuron)
    float max_a = 1e-8f;
    for (int i = 0; i < n_in; i++)
        if (inputs[i] > max_a) max_a = inputs[i];

    float max_w = 1e-8f;
    for (int j = 0; j < n_out; j++)
        for (int i = 0; i < n_in; i++) {
            float wf = fabsf((float)(int8_t)pgm_read_byte(
                &weights[j * n_in + i]) / weight_scale);
            if (wf > max_w) max_w = wf;
        }

    for (int j = 0; j < n_out; j++) {
        vTaskDelay(0);   // yield to RTOS watchdog
        float acc = 0.0f;
        for (int i = 0; i < n_in; i++) {
            uint8_t px    = (uint8_t)((inputs[i] / max_a) * 255.0f);
            int8_t  raw_w = (int8_t)pgm_read_byte(&weights[j * n_in + i]);
            float   wf    = (float)raw_w / weight_scale;
            uint8_t pw    = (uint8_t)((fabsf(wf) / max_w) * 255.0f);

            uint32_t sr   = sc_word(px) & sc_word(pw);
            float product = (float)__builtin_popcount(sr) / 32.0f;
            acc += (wf >= 0.0f ? 1.0f : -1.0f) * product;
        }
        int8_t b  = (int8_t)pgm_read_byte(&biases[j]);
        float  val = acc + (float)b / bias_scale;
        outputs[j] = val > 0.0f ? val : 0.0f;   // ReLU
    }
}

// ---------------------------------------------------------------------------
// sc_argmax
// ---------------------------------------------------------------------------
int sc_argmax(const float* scores, int n) {
    float best = -1e9f;
    int   cls  = 0;
    for (int i = 0; i < n; i++) {
        if (scores[i] > best) { best = scores[i]; cls = i; }
    }
    return cls;
}
