/*
 * tflm_inference_nir.cpp
 * =======================
 * TensorFlow Lite Micro inference energy benchmark for ESP32 — NIR 3-class.
 *
 * Identical structure to Thesis-Edge-AI tflm_inference.cpp.
 * Only label-related changes: Logistic (Sigmoid) -> Softmax op,
 * N_OUTPUTS 1 -> 3, threshold -> argmax prediction.
 *
 * Loads a pre-quantised INT8 TFLite model, runs a forward pass on every test
 * sample, and reports accuracy + per-inference timing.
 *
 * Architecture: Input(10) -> Dense(16, ReLU) -> Dense(3, Softmax)
 * Data type:    INT8 (quantised internally); float32 I/O
 * Framework:    TF Lite Micro via EloquentTinyML v3 + tflm_esp32
 * Labels:       0=Fresh  1=Aging  2=Degraded
 *
 * PPK2 measurement protocol:
 *   - Built-in LED (GPIO2) is HIGH during the benchmark window only
 *   - Start PPK2 when LED turns on, stop when LED turns off
 *   - Energy/inference = total_energy_uJ / (N_REPEATS * N_TEST_NIR)
 *
 * Headers required:
 *   tflm_model_nir.h                -> g_tflm_model_nir, g_tflm_model_nir_len
 *   mould_prediction_dataset_nir.h  -> nir_test_X, nir_test_y, N_TEST_NIR
 *
 * Dependencies (platformio.ini env:tflm_inference_nir):
 *   https://github.com/eloquentarduino/tflm_esp32
 *   https://github.com/eloquentarduino/EloquentTinyML
 */

#include <Arduino.h>

#include "tflm_model_nir.h"                // g_tflm_model_nir, g_tflm_model_nir_len
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

#include "mould_prediction_dataset_nir.h"  // nir_test_X, nir_test_y, N_TEST_NIR

// ---------------------------------------------------------------------------
// Benchmark settings
// ---------------------------------------------------------------------------
#define N_REPEATS     100
#define PRINT_PREDS   false
#define LED_PIN       2

// ---------------------------------------------------------------------------
// Network dimensions
// ---------------------------------------------------------------------------
#define N_INPUTS    10
#define N_OUTPUTS   3   // 3-class softmax output

// Ops used: FullyConnected (Dense), Relu, Softmax, Quantize, Dequantize
// 10 KB arena — slightly more than binary model needs due to 3-output layer
#define ARENA_SIZE  (10 * 1024)
#define NUM_OPS     5

Eloquent::TF::Sequential<NUM_OPS, ARENA_SIZE> tf;

static const char* CLASS_NAMES[3] = {"Fresh", "Aging", "Degraded"};

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
void setup() {
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    Serial.begin(115200);
    while (!Serial) delay(10);

    Serial.println("\n========================================");
    Serial.println("  TF Lite Micro Inference Benchmark — NIR 3-Class");
    Serial.println("  Architecture: 10 -> Dense(16,ReLU) -> Dense(3,Softmax)");
    Serial.println("  Data type: INT8 (quantised), float32 I/O");
    Serial.println("  Library: EloquentTinyML v3 + tflm_esp32");
    Serial.println("  Labels: Fresh / Aging / Degraded");
    Serial.println("========================================");
    Serial.printf("  Model size   : %u bytes (%.1f KB)\n",
                  g_tflm_model_nir_len, g_tflm_model_nir_len / 1024.0f);
    Serial.printf("  Test samples : %d\n", N_TEST_NIR);
    Serial.printf("  Repeats      : %d\n", N_REPEATS);
    Serial.printf("  Total inferences: %d\n", N_TEST_NIR * N_REPEATS);

    Serial.println("\nInitialising TFLite Micro interpreter...");

    tf.setNumInputs(N_INPUTS);
    tf.setNumOutputs(N_OUTPUTS);
    tf.resolver.AddFullyConnected();
    tf.resolver.AddRelu();
    tf.resolver.AddSoftmax();   // 3-class softmax (replaces Logistic/Sigmoid from binary model)
    tf.resolver.AddQuantize();
    tf.resolver.AddDequantize();

    while (!tf.begin(g_tflm_model_nir).isOk()) {
        Serial.print("FATAL: TFLM init failed: ");
        Serial.println(tf.exception.toString());
        delay(3000);
    }

    Serial.println("Interpreter ready.");
    Serial.printf("  Arena used: %u / %u bytes\n",
                  tf.interpreter->arena_used_bytes(), ARENA_SIZE);
    Serial.println("Starting in 2 seconds...");
    delay(2000);

    // -----------------------------------------------------------------------
    // BENCHMARK START -- LED turns ON
    // -----------------------------------------------------------------------
    digitalWrite(LED_PIN, HIGH);
    Serial.println("\n=== BENCHMARK START ===");

    uint32_t class_correct[3] = {0, 0, 0};
    uint32_t class_total[3]   = {0, 0, 0};
    uint32_t total_correct    = 0;
    uint32_t total_samples    = 0;
    uint32_t t_start          = micros();

    for (int rep = 0; rep < N_REPEATS; rep++) {
        for (int i = 0; i < N_TEST_NIR; i++) {
            float input_buf[N_INPUTS];
            memcpy(input_buf, nir_test_X[i], N_INPUTS * sizeof(float));

            if (!tf.predict(input_buf).isOk()) {
                Serial.println("ERROR: predict() failed");
                continue;
            }

            // Argmax over 3 softmax outputs
            uint8_t pred = 0;
            for (uint8_t c = 1; c < N_OUTPUTS; c++) {
                if (tf.output(c) > tf.output(pred)) pred = c;
            }
            uint8_t actual = nir_test_y[i];

            if (pred == actual) {
                total_correct++;
                if (rep == 0) class_correct[actual]++;
            }
            if (rep == 0) class_total[actual]++;
            total_samples++;

#if PRINT_PREDS
            if (rep == 0) {
                Serial.printf("  [%3d] pred=%s actual=%s probs=[%.3f %.3f %.3f] %s\n",
                              i, CLASS_NAMES[pred], CLASS_NAMES[actual],
                              tf.output(0), tf.output(1), tf.output(2),
                              pred == actual ? "OK" : "WRONG");
            }
#endif
        }
    }

    uint32_t t_end = micros();
    // -----------------------------------------------------------------------
    // BENCHMARK END -- LED turns OFF
    // -----------------------------------------------------------------------
    digitalWrite(LED_PIN, LOW);
    Serial.println("=== BENCHMARK END ===\n");

    uint32_t elapsed_us       = t_end - t_start;
    float    us_per_inference = (float)elapsed_us / (float)total_samples;
    float    ms_per_inference = us_per_inference / 1000.0f;
    float    accuracy         = 100.0f * total_correct / total_samples;

    Serial.println("--- Results ---");
    Serial.printf("  Total inferences  : %u\n", total_samples);
    Serial.printf("  Correct           : %u\n", total_correct);
    Serial.printf("  Overall accuracy  : %.1f%%\n", accuracy);
    Serial.println("  Per-class accuracy (first repeat only):");
    for (int c = 0; c < 3; c++) {
        if (class_total[c] > 0) {
            Serial.printf("    %-10s : %.1f%%  (%u / %u)\n",
                          CLASS_NAMES[c],
                          100.0f * class_correct[c] / class_total[c],
                          class_correct[c], class_total[c]);
        }
    }
    Serial.printf("  Total time        : %u us (%.2f ms)\n", elapsed_us, elapsed_us / 1000.0f);
    Serial.printf("  Time/inference    : %.1f us (%.3f ms)\n", us_per_inference, ms_per_inference);
    Serial.println("\nRecord PPK2 energy between BENCHMARK START and END.");
    Serial.println("Energy/inference = total_energy_uJ / total_inferences");
}

void loop() {
    delay(10000);
}
