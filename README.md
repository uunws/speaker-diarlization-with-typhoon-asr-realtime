# 🌪️ Typhoon ASR Real-time 

![Model Size: 115M](https://img.shields.io/badge/Model%20Size-115M-blue) ![Pretrained Data: 11k Hours](https://img.shields.io/badge/Pretrained%20Data-11%2C000%20Hours-green) 

A **115M-parameter FastConformer-Transducer** model designed for low-latency Thai speech recognition, pretrained on **11,000 hours of Thai audio** (public and internal datasets).

Inspired by the need for real-time streaming capabilities in production for morphologically complex and low-resource languages like Thai, this model prioritizes low latency, consistency, and hallucination-free transcription over generic offline models.

---

## 🏗️ Architecture

Instead of relying on large-scale models like Whisper, Typhoon ASR Real-time focuses on speed and efficiency for practical usage:

- **Fast Conformer-Transducer:** Utilizes a structure that supports real-time streaming (processes audio and transcribes instantly), unlike Whisper which processes in chunks and introduces latency.
- **High Efficiency:** Contains only **115 million parameters** (13x smaller than Whisper Large-v3) and reduces computation (GFLOPs) by up to 45x.
- **Aggressive Downsampling:** Uses downsampling in the Encoder, processing audio **2.4x faster** than a standard Conformer.

---

## 📊 Data Strategy & Normalization

The core principle behind this model is **Training Data Quality and Consistency**:

- **Text Normalization:** Employs strict rules to standardize text. For example, contextual reading of sequences like postal codes vs. large numbers (e.g., "10150" becomes *“หนึ่งศูนย์หนึ่งห้าศูนย์”* to avoid *“หนึ่งหมื่น一百ห้าสิบ”*). Handled smoothly with an **internal TTS (Orpheus)** for complex numeric sequences.
- **Consensus-Based Labeling:** Uses a majority voting system from 3 Whisper models for pseudo-labeling to filter 11,000 hours of training data.
- **Human-in-the-loop:** Complex texts with numbers or special symbols are verified and corrected by human evaluators.

---

## 🇹🇭 Isan Dialect Adaptation

The Isan dialect model is built upon the base ASR, utilizing **Curriculum Learning** to understand the dialect without forgetting central Thai:

1. **Stage 1 (Global Adaptation):** Full-model fine-tuning for 10 epochs at a low learning rate (`η = 10⁻⁵`). Gradually adapts the acoustic filter banks in the Encoder to recognize specific Isan tones without destroying the features learned during general pretraining.
2. **Stage 2 (Linguistic Tuning):** Freezes the Encoder and fine-tunes the Decoder and Joint Network for 15 epochs at a higher learning rate (`η = 10⁻³`). Focuses on Isan-specific vocabulary, grammar, and particles (e.g., "บ่", "เฮ็ด") using stable audio signals from Stage 1. 
   - *Result:* Stage 2 tuning drops the Character Error Rate (CER) by an additional 5.57%.

---

## ⚙️ Under the Hood

### Training Details
- **Hardware Efficiency:** Initialized from `nvidia/stt_en_fastconformer_transducer_large`. 1 epoch takes just **17 hours** on 2×NVIDIA H100 GPUs.
- **Optimizer:** `AdamW` (with weight decay to reduce loss and overfitting) + `Cosine Annealing` (reduces learning rate gradually). Includes 5,000 Warmup steps.

### Technology Stack
- **NVIDIA NeMo:** An open-source toolkit for Conversational AI. It provides the Fast Conformer architecture. NeMo seamlessly handles distributing tasks across GPUs.
- **CUDA:** Processes complex computing tasks in parallel using the GPU, significantly speeding up Deep Learning training.
- **Warp RNN-T Numba:** A highly optimized Loss function tailored for high-speed calculation.
  - **Warp-CTC:** Solves alignment problems in speech (matching audio frames to text lengths without manual mapping).
  - **RNN-Transducer (RNN-T):** Consists of an Encoder, Predictor, and Joiner. Excellent for assessing if the current sound matches the previously spoken word, highly suited for real-time.
  - **Numba (JIT):** Replaces the need for complex C++/CUDA code by compiling Python to run fast on GPUs natively.

---

## 🏆 Benchmarks & Evaluation

Evaluated across two main tracks (Typhoon vs Base Models):

- **Standard Track (Gigaspeech2):** Reduced CER from 5.84% to **4.69%**. Gigaspeech2 involves 10,000 refined hours of Thai text.
- **Robustness Track (TVSpeech):** Tested on noisy, complex YouTube audio. Reduced CER from 10.36% to **6.32%**.

*Note: While massive models like Gemini 1.5 Pro excel at comprehension, Typhoon performs better in verbatim transcription for Thai due to its focused training payload.*

### Model Family Comparison

| Detail | Typhoon ASR Real-time | Typhoon ASR (Offline) | Typhoon Isan Real-time |
| :--- | :--- | :--- | :--- |
| **Usage** | Streaming | Offline | Streaming |
| **Architecture** | Fast Conformer-Transducer | Whisper-based | Fast Conformer-Transducer |
| **Size** | 115M | 1,550M (Large-v3) | 115M |
| **Accuracy (CER)** | 9.99% | 6.32% (Highest Accuracy) | 9.34% |
| **Efficiency** | Extremely fast (45x lighter) | High Resource | Extremely fast (45x lighter) |

---

## 🌐 Available Models (Version 1)

- **ASR Real-time:** [typhoon-asr-realtime](https://opentyphoon.ai/model/typhoon-asr-realtime) *(8 Sep 2025)*
- **Isan ASR Whisper:** [typhoon-isan-asr-whisper](https://opentyphoon.ai/model/typhoon-isan-asr-whisper) *(27 Nov 2025)*
- **Isan ASR Real-time:** [typhoon-isan-asr-realtime](https://opentyphoon.ai/model/typhoon-isan-asr-realtime) *(27 Nov 2025)*

---

## 🚧 Limitations & Future Work

- **Formatting Constraints:** Due to strict verbatim transcription, the output is raw text (e.g., writing "สิบ" instead of "10"). This requires an Inverse Text Normalization (ITN) module downstream for readability.
- **Code-Switching:** Currently limited when handling Thai-English mixed speech; often transliterates English into Thai phrasing.
- **Future Directions:** Expansion to Northern and Southern dialects and further model compression for on-device deployment.

---

