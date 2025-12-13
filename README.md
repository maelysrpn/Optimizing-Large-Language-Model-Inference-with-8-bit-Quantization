# Optimizing-Large-Language-Model-Inference-with-8-bit-Quantization

## Project Overview
This project aims to optimize the inference process of Large Language Models (LLMs), specifically the **LLaMA-2 7B** architecture. 
The primary goal is to implement and analyze **8-bit quantization** techniques to drastically reduce the model's memory footprint (VRAM) while maintaining text generation quality.

This work relies on the `transformers` ecosystem and the `bitsandbytes` library, implementing the vector-wise quantization strategy described in the *LLM.int8()* paper.

## Objectives
1.  **Understand** the Causal Transformer architecture and LLaMA specificities (RoPE, RMSNorm, SwiGLU).
2.  **Implement** 8-bit quantization using mixed-precision decomposition.
3.  **Benchmark** the performance: compare the quantized model against the FP16 baseline in terms of Memory, Latency, and Quality.

## Methodology

### Model Architecture
We utilize **NousResearch/Llama-2-7b-chat-hf**, a fine-tuned version of LLaMA 2.
- **Parameters:** 7 Billion.
- **Precision:** We compare **FP16** (Half-Precision) vs **INT8** (Quantized).

### Quantization Strategy
We use the **Mixed-Precision Decomposition** technique, as presented in the LLM.int(8) paper:
- **Outliers:** Feature dimensions with values $> 6.0$ are preserved in **FP16** to maintain model stability and reasoning capabilities.
- **Main Body:** 99.9% of the weights are compressed to **8-bit integers**.

## Performance Analysis (Key Results)

Experiments were conducted on **Google Colab** using a **NVIDIA Tesla T4 GPU (15GB VRAM)**.

| Metric | Baseline (FP16) | Quantized (8-bit) | Improvement / Impact |
| :--- | :--- | :--- | :--- |
| **VRAM Usage** | **12.55 GB** | **6.53 GB** | **-48% Memory Footprint** |
| **Inference Speed** | 16.58 tokens/s | 6.77 tokens/s | Higher latency due to dequantization overhead on T4 |
| **Text Quality** | High | High | No noticeable degradation |

**Key Findings:**
1.  **Memory Efficiency:** Quantization successfully allows the 7B model to fit on smaller GPUs (8GB VRAM cards), whereas the FP16 version requires strictly >12GB.
2.  **Quality Preservation:** The "Outlier" threshold technique proved effective; the generated travel plans for Paris were coherent and factually correct in both versions, not talking about visiting Trafalgare Square or other hallucinations.
3.  **Speed Trade-off:** On older hardware like the T4, 8-bit inference is slower due to the compute cost of dequantizing values on the fly. This gap would be smaller on modern Ampere/Hopper GPUs.

## How to Run

### Prerequisites
- Python 3.8+
- A GPU with at least 8GB of VRAM (for 8-bit) or 16GB (for FP16).

### Installation
```bash
pip install transformers accelerate bitsandbytes scipy torch
