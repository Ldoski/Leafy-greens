/*
 * tinyol_benchmark.cpp — v8 (3-class softmax, NIR labels)
 * =====================
 * TinyOL-style on-device learning benchmark for ESP32.
 *
 * Implements the core concept from Ren et al. (2021) "TinyOL: TinyML with
 * Online-Learning on Microcontrollers" (arXiv:2103.08295):
 *
 *   Philosophy: Freeze the pre-trained feature extractor so the device only
 *   needs to adapt the final output layer. This dramatically reduces the cost
 *   of on-device learning — only 51 parameters are updated per sample instead
 *   of all 227 weights.
 *
 * Architecture:
 *   Input(10) -> Dense(16, ReLU) [FROZEN]  -> Dense(3, Softmax) [TRAINABLE]
 *
 * Weight source: tinyol_weights.h — a WEAK backbone trained on Batch 1 ONLY
 *   (high-temp data, ~250 samples). This is intentionally weaker than the full
 *   AIfES backbone (aifes_weights_nir.h) to create a genuine accuracy gap that
 *   TinyOL's on-device fine-tuning can close.
 *
 *   With the full backbone, TinyOL had no effect: pre-trained weights already
 *   generalised well, leaving 0% improvement headroom.
 *   With the Batch-1-only backbone, the model is weaker on cold-storage data —
 *   Batch 4 fine-tuning adapts the output layer to that new environment.
 *
 * Data split (Option A — authentic weak backbone):
 *   - Backbone trained on PC: Batch 1 ONLY (fit) + Batch 2 (val, early-stop)
 *   - Batches 3+4 withheld from backbone entirely
 *   - On-device fine-tuning:  Batch 4 (held_out_dataset.h)
 *   - Evaluation:             Batch 3 (mould_prediction_dataset_nir.h, never
 *                             seen during any part of the training pipeline)
 *
 * Training:
 *   - Loss:          Weighted Cross-Entropy (3-class) for class imbalance
 *   - Optimizer:     Stochastic Gradient Descent (no momentum)
 *   - LR:            0.001  (simulation-validated with class weights + 10 epochs)
 *   - Epochs:        10  (multi-pass over held-out buffer; TinyOL allows replay)
 *   - Class weights: w[c] = N_HELD / (N_CLASSES * N_HELD_CLASS[c])
 *                    Inverse frequency — balances gradient across 3 classes.
 *   - Shuffling:     Fisher-Yates shuffle (esp_random() TRNG, no seed)
 *
 * Version history:
 *   v1-v7: binary sigmoid (binary mould/no-mould labels)
 *   v8:    updated to 3-class softmax to match tinyol_weights.h (NIR 3-class)
 *          Removed THRESHOLD (argmax replaces threshold comparison).
 *          W_CLASS_POS/NEG replaced by W_CLASS[3] per-class weights.
 *          W_o resized from [16] to [48] (16 hidden * 3 outputs).
 *          B_o changed from scalar to B_o[3].
 *          forward() now fills logits[] and probs[] buffers.
 *          backward() updates all 3 output units via cross-entropy gradient.
 *          Include changed to mould_prediction_dataset_nir.h.
 *          Test variables: test_X/test_y/N_TEST -> nir_test_X/nir_test_y/N_TEST_NIR.
 *
 * Measurement protocol (identical to aifes_inference and tflm_inference):
 *   - LED GPIO2 HIGH during benchmark window (training loop only)
 *   - RAM stats printed before/after the window (outside PPK2 window)
 *   - Accuracy measured before and after training (outside PPK2 window)
 *   - Energy/update = total_energy_uJ / (N_EPOCHS * N_HELD)
 *   - One "update" = one forward pass + one backward pass on one sample
 *
 * No external libraries required — pure C++ using Arduino math.h + esp_random().
 */

#include <Arduino.h>
#include <math.h>

#include "tinyol_weights.h"                // aifes_flat_weights[], AIFES_NIR_*_SIZE — Batch1-only backbone
#include "mould_prediction_dataset_nir.h"  // nir_test_X, nir_test_y, N_TEST_NIR, N_FEATURES — evaluation only
#include "held_out_dataset.h"              // held_X, held_y, N_HELD, N_HELD_FRESH/AGING/DEGRADED — Batch 4

// ---------------------------------------------------------------------------
// Benchmark settings
// ---------------------------------------------------------------------------
#define N_EPOCHS      10       // Multi-pass over Batch 4 held-out buffer.
                               // Class weights prevent majority-class collapse.
#define LR            0.001f  // SGD LR — simulation-validated with class weights.
#define LED_PIN       2        // Built-in LED: ON during benchmark, OFF before/after
#define PRINT_LOSS    false    // Set true to print per-epoch loss (slows benchmark)

// ---------------------------------------------------------------------------
// Class weights for held-out batch imbalance.
// w[c] = N_HELD / (N_CLASSES * count[c]) — inverse frequency per class.
// Computed at runtime from the header-defined constants so they automatically
// update when prepare_dataset_NIR.py regenerates held_out_dataset.h.
// ---------------------------------------------------------------------------
static float W_CLASS[AIFES_NIR_OUTPUT_SIZE];

// ---------------------------------------------------------------------------
// Weight layout in aifes_flat_weights[] (from tinyol_weights.h):
//
//   W1: indices [0 .. 159]   — Dense(10->16), row-major (k=input, j=hidden)
//       W1[k][j] = aifes_flat_weights[k * AIFES_NIR_HIDDEN_SIZE + j]
//
//   B1: indices [160 .. 175] — biases for hidden layer, B1[j] for j = 0..15
//
//   W2: indices [176 .. 223] — Dense(16->3), row-major (j=hidden, c=class)
//       W2[j][c] = aifes_flat_weights[176 + j * AIFES_NIR_OUTPUT_SIZE + c]
//
//   B2: indices [224 .. 226] — 3 output biases, one per class
//
//   Total: 160+16+48+3 = 227  (matches AIFES_NIR_N_WEIGHTS)
// ---------------------------------------------------------------------------

// Frozen feature extractor — const pointers into flash, no copy needed
static const float* W_h = &aifes_flat_weights[0];    // 160 floats  W1
static const float* B_h = &aifes_flat_weights[160];  //  16 floats  B1

// Trainable output layer — copied to RAM so SGD can modify them
static float W_o[AIFES_NIR_HIDDEN_SIZE * AIFES_NIR_OUTPUT_SIZE];  // 48 floats  W2
static float B_o[AIFES_NIR_OUTPUT_SIZE];                           //  3 floats  B2

// Shuffle index array over Batch 4 training samples
static int shuffle_idx[N_HELD];

// ---------------------------------------------------------------------------
// Initialise output layer from pre-trained backbone values.
// ---------------------------------------------------------------------------
static void initOutputLayer() {
    memcpy(W_o, &aifes_flat_weights[176], AIFES_NIR_HIDDEN_SIZE * AIFES_NIR_OUTPUT_SIZE * sizeof(float));
    memcpy(B_o, &aifes_flat_weights[224], AIFES_NIR_OUTPUT_SIZE * sizeof(float));
}

// ---------------------------------------------------------------------------
// Fisher-Yates shuffle of shuffle_idx[].
// Uses esp_random() — hardware TRNG on ESP32, no seed needed.
// ---------------------------------------------------------------------------
static void shuffleIndices() {
    for (int i = N_HELD - 1; i > 0; i--) {
        int j = (int)(esp_random() % (uint32_t)(i + 1));
        int tmp = shuffle_idx[i];
        shuffle_idx[i] = shuffle_idx[j];
        shuffle_idx[j] = tmp;
    }
}

// ---------------------------------------------------------------------------
// Forward pass
// Fills hidden[] (16 ReLU activations), logits[] and probs[] (3 softmax outputs).
// Caller supplies all three buffers. logits[] and probs[] are needed by
// backward() — pass the same arrays through to avoid re-allocating.
// Numeric-stable softmax: subtract max before exp() to prevent overflow.
// ---------------------------------------------------------------------------
static void forward(const float* input, float* hidden, float* logits, float* probs) {
    // Feature extractor — Dense(10->16, ReLU), frozen
    for (int j = 0; j < AIFES_NIR_HIDDEN_SIZE; j++) {
        float sum = B_h[j];
        for (int k = 0; k < AIFES_NIR_INPUT_SIZE; k++) {
            sum += W_h[k * AIFES_NIR_HIDDEN_SIZE + j] * input[k];
        }
        hidden[j] = (sum > 0.0f) ? sum : 0.0f;  // ReLU
    }

    // Output layer — Dense(16->3), trainable, produces 3 raw logits
    for (int c = 0; c < AIFES_NIR_OUTPUT_SIZE; c++) {
        float sum = B_o[c];
        for (int j = 0; j < AIFES_NIR_HIDDEN_SIZE; j++) {
            sum += W_o[j * AIFES_NIR_OUTPUT_SIZE + c] * hidden[j];
        }
        logits[c] = sum;
    }

    // Softmax (numerically stable)
    float max_l = logits[0];
    for (int c = 1; c < AIFES_NIR_OUTPUT_SIZE; c++) {
        if (logits[c] > max_l) max_l = logits[c];
    }
    float sum_exp = 0.0f;
    for (int c = 0; c < AIFES_NIR_OUTPUT_SIZE; c++) {
        probs[c] = expf(logits[c] - max_l);
        sum_exp += probs[c];
    }
    for (int c = 0; c < AIFES_NIR_OUTPUT_SIZE; c++) {
        probs[c] /= sum_exp;
    }
}

// ---------------------------------------------------------------------------
// Backward pass + SGD update (output layer only)
//
// Gradient of weighted cross-entropy w.r.t. logit_c (before softmax):
//   delta_c = W_CLASS[label] * (probs[c] - (c == label ? 1.0f : 0.0f))
//
// This is the standard softmax + cross-entropy gradient scaled by the
// class weight for the true label.  Only the weight for the correct class
// applies — incorrect-class gradients already have their natural magnitude
// from the softmax denominator.
//
// Weight gradient:  dL/dW_o[j][c] = delta_c * hidden[j]
// Bias gradient:    dL/dB_o[c]    = delta_c
// SGD update:       param -= LR * gradient
// ---------------------------------------------------------------------------
static void backward(const float* hidden, const float* probs, uint8_t label) {
    float w = W_CLASS[label];
    for (int c = 0; c < AIFES_NIR_OUTPUT_SIZE; c++) {
        float delta = w * (probs[c] - (c == (int)label ? 1.0f : 0.0f));
        for (int j = 0; j < AIFES_NIR_HIDDEN_SIZE; j++) {
            W_o[j * AIFES_NIR_OUTPUT_SIZE + c] -= LR * delta * hidden[j];
        }
        B_o[c] -= LR * delta;
    }
}

// ---------------------------------------------------------------------------
// Evaluate accuracy over the full test set using current weights.
// Prediction = argmax of the 3 softmax probabilities (no threshold).
// ---------------------------------------------------------------------------
static uint32_t evaluateAccuracy() {
    float hidden[AIFES_NIR_HIDDEN_SIZE];
    float logits[AIFES_NIR_OUTPUT_SIZE];
    float probs[AIFES_NIR_OUTPUT_SIZE];
    uint32_t correct = 0;
    for (int i = 0; i < N_TEST_NIR; i++) {
        forward(nir_test_X[i], hidden, logits, probs);
        uint8_t pred = 0;
        for (uint8_t c = 1; c < AIFES_NIR_OUTPUT_SIZE; c++) {
            if (probs[c] > probs[pred]) pred = c;
        }
        if (pred == nir_test_y[i]) correct++;
    }
    return correct;
}

// ---------------------------------------------------------------------------
// Setup — runs once, contains entire benchmark
// ---------------------------------------------------------------------------
void setup() {
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    Serial.begin(115200);
    while (!Serial) delay(10);

    Serial.println("\n========================================");
    Serial.println("  TinyOL On-Device Learning Benchmark  [v8]");
    Serial.println("  Frozen:    10 -> Dense(16, ReLU)     [pre-trained]");
    Serial.println("  Trainable:      Dense(3,  Softmax)   [SGD on-device]");
    Serial.println("  Method: TinyOL (Ren et al. 2021, arXiv:2103.08295)");
    Serial.println("  Loss: Weighted Cross-Entropy  |  Optimizer: SGD  |  10 epochs");
    Serial.println("========================================");
    Serial.printf("  Total weights       : %d floats\n",  AIFES_NIR_N_WEIGHTS);
    Serial.printf("  Frozen params       : %d  (W1[160] + B1[16])\n",
                  AIFES_NIR_INPUT_SIZE * AIFES_NIR_HIDDEN_SIZE + AIFES_NIR_HIDDEN_SIZE);
    Serial.printf("  Trainable params    : %d  (W2[48] + B2[3])\n",
                  AIFES_NIR_HIDDEN_SIZE * AIFES_NIR_OUTPUT_SIZE + AIFES_NIR_OUTPUT_SIZE);
    Serial.printf("  Training data       : Batch 4 (held_out) — %d samples\n", N_HELD);
    Serial.printf("    Fresh=%d  Aging=%d  Degraded=%d\n",
                  N_HELD_FRESH, N_HELD_AGING, N_HELD_DEGRADED);
    Serial.printf("  Evaluation data     : Batch 3 (test)     — %d samples\n", N_TEST_NIR);
    Serial.printf("  Epochs              : %d  (multi-pass over held-out buffer)\n", N_EPOCHS);
    Serial.printf("  Total updates       : %d  (%d epochs x %d samples)\n",
                  N_HELD * N_EPOCHS, N_EPOCHS, N_HELD);
    Serial.printf("  Learning rate       : %.5f  (SGD, no momentum)\n", LR);

    // Compute per-class weights from held-out distribution (runtime, not hardcoded).
    // Inverse frequency: w[c] = N_HELD / (N_CLASSES * count[c])
    W_CLASS[0] = (float)N_HELD / (3.0f * (float)N_HELD_FRESH);
    W_CLASS[1] = (float)N_HELD / (3.0f * (float)N_HELD_AGING);
    W_CLASS[2] = (float)N_HELD / (3.0f * (float)N_HELD_DEGRADED);

    // Initialise shuffle index array over Batch 4 training samples
    for (int i = 0; i < N_HELD; i++) shuffle_idx[i] = i;

    initOutputLayer();
    Serial.println("\nWeights loaded. Output layer initialised from pre-trained values.");
    Serial.printf("  Class weight Fresh  : %.4f\n", W_CLASS[0]);
    Serial.printf("  Class weight Aging  : %.4f\n", W_CLASS[1]);
    Serial.printf("  Class weight Degrad : %.4f\n", W_CLASS[2]);

    // -----------------------------------------------------------------------
    // Accuracy BEFORE training (outside PPK2 window)
    // -----------------------------------------------------------------------
    uint32_t correct_before = evaluateAccuracy();
    Serial.printf("\nAccuracy BEFORE training : %.1f%%  (%u / %d)\n",
                  100.0f * correct_before / N_TEST_NIR, correct_before, N_TEST_NIR);

    // -----------------------------------------------------------------------
    // RAM snapshot BEFORE benchmark (outside PPK2 window)
    // -----------------------------------------------------------------------
    uint32_t heap_total  = ESP.getHeapSize();
    uint32_t heap_before = ESP.getFreeHeap();
    uint32_t heap_used   = heap_total - heap_before;
    Serial.println("\n--- Memory (before benchmark) ---");
    Serial.printf("  Heap total         : %6u B  (%4.1f KB)\n", heap_total,  heap_total  / 1024.0f);
    Serial.printf("  Heap free          : %6u B  (%4.1f KB)\n", heap_before, heap_before / 1024.0f);
    Serial.printf("  Heap used          : %6u B  (%4.1f KB)\n", heap_used,   heap_used   / 1024.0f);
    Serial.printf("  Trainable params   : %6u B  (%4.1f KB)  [BSS/stack, not heap]\n",
                  (AIFES_NIR_HIDDEN_SIZE * AIFES_NIR_OUTPUT_SIZE + AIFES_NIR_OUTPUT_SIZE) * 4,
                  (AIFES_NIR_HIDDEN_SIZE * AIFES_NIR_OUTPUT_SIZE + AIFES_NIR_OUTPUT_SIZE) * 4 / 1024.0f);
    Serial.printf("  Shuffle index buf  : %6u B  (%4.1f KB)  [BSS, not heap]\n",
                  N_HELD * 4, N_HELD * 4 / 1024.0f);
    Serial.printf("  Frozen W+B in flash: %6u B  (%4.1f KB)  [static const, not heap]\n",
                  AIFES_NIR_N_WEIGHTS * 4, AIFES_NIR_N_WEIGHTS * 4 / 1024.0f);

    Serial.println("\nStarting in 2 seconds... (start PPK2 now)");
    delay(2000);

    // -----------------------------------------------------------------------
    // BENCHMARK START -- LED turns ON
    // -----------------------------------------------------------------------
    digitalWrite(LED_PIN, HIGH);
    Serial.println("\n=== BENCHMARK START ===");

    uint32_t t_start = micros();

    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;

        shuffleIndices();

        for (int i = 0; i < N_HELD; i++) {
            int s = shuffle_idx[i];
            float hidden[AIFES_NIR_HIDDEN_SIZE];
            float logits[AIFES_NIR_OUTPUT_SIZE];
            float probs[AIFES_NIR_OUTPUT_SIZE];

            forward(held_X[s], hidden, logits, probs);

            // Cross-entropy loss: -log(prob of correct class)
            float eps = 1e-7f;
            epoch_loss -= logf(probs[(int)held_y[s]] + eps);

            backward(hidden, probs, held_y[s]);
        }

#if PRINT_LOSS
        Serial.printf("  Epoch %2d/%d  avg_loss=%.4f\n",
                      epoch + 1, N_EPOCHS, epoch_loss / N_HELD);
#endif
    }

    uint32_t t_end = micros();
    // -----------------------------------------------------------------------
    // BENCHMARK END -- LED turns OFF
    // -----------------------------------------------------------------------
    digitalWrite(LED_PIN, LOW);
    Serial.println("=== BENCHMARK END ===\n");

    // -----------------------------------------------------------------------
    // RAM snapshot AFTER benchmark
    // -----------------------------------------------------------------------
    uint32_t heap_after = ESP.getFreeHeap();
    uint32_t min_heap   = ESP.getMinFreeHeap();
    Serial.println("--- Memory (after benchmark) ---");
    Serial.printf("  Heap free after    : %6u B  (%4.1f KB)\n", heap_after, heap_after / 1024.0f);
    Serial.printf("  Min free (peak)    : %6u B  (%4.1f KB)\n", min_heap,   min_heap   / 1024.0f);
    Serial.printf("  Peak heap used     : %6u B  (%4.1f KB)\n",
                  heap_total - min_heap, (heap_total - min_heap) / 1024.0f);
    Serial.printf("  Heap leak          : %6d B  (before-after; 0 expected)\n",
                  (int)heap_before - (int)heap_after);

    // -----------------------------------------------------------------------
    // Accuracy AFTER training (outside PPK2 window)
    // -----------------------------------------------------------------------
    uint32_t correct_after = evaluateAccuracy();
    Serial.printf("\nAccuracy AFTER training  : %.1f%%  (%u / %d)\n",
                  100.0f * correct_after / N_TEST_NIR, correct_after, N_TEST_NIR);

    // -----------------------------------------------------------------------
    // Results
    // One "update" = one forward pass + one backward pass on one sample.
    // Directly comparable to aifes_inference and tflm_inference energy numbers,
    // but includes the gradient computation and weight update cost.
    // -----------------------------------------------------------------------
    uint32_t total_updates     = (uint32_t)N_HELD * N_EPOCHS;
    uint32_t elapsed_us        = t_end - t_start;
    float    us_per_update     = (float)elapsed_us / (float)total_updates;
    float    ms_per_update     = us_per_update / 1000.0f;
    float    accuracy_before   = 100.0f * (float)correct_before / (float)N_TEST_NIR;
    float    accuracy_after    = 100.0f * (float)correct_after  / (float)N_TEST_NIR;
    uint32_t cycles_per_update = (uint32_t)(us_per_update * 240.0f);  // at 240 MHz

    Serial.println("\n--- Results ---");
    Serial.printf("  Accuracy before   : %.1f%%\n", accuracy_before);
    Serial.printf("  Accuracy after    : %.1f%%  (delta: %+.1f%%)\n",
                  accuracy_after, accuracy_after - accuracy_before);
    Serial.printf("  Total updates     : %u  (%d epochs x %d held-out samples)\n",
                  total_updates, N_EPOCHS, N_HELD);
    Serial.printf("  Total time        : %u us (%.2f ms)\n",
                  elapsed_us, elapsed_us / 1000.0f);
    Serial.printf("  Time/update       : %.1f us (%.3f ms)\n",
                  us_per_update, ms_per_update);
    Serial.printf("  CPU cycles/update : ~%u cycles  (at 240 MHz)\n",
                  cycles_per_update);
    Serial.println("\nRecord PPK2 energy between BENCHMARK START and END.");
    Serial.println("Energy/update = total_energy_uJ / total_updates");
    Serial.println("Compare to AIfES inference: energy/inference (forward-pass only)");
}

void loop() {
    delay(10000);
}
