# Setup & Exploration Report: FinGPT Sentiment Analysis

## 1. Exploration Overview
The FinGPT project (AI4Finance-Foundation/FinGPT) provides several versions of financial large language models. The sentiment analysis task specifically leverages instruction-tuning on financial datasets.

### Core Inference Entry Points Identified:
- **v1/v2:** Typically based on BERT/RoBERTa variants.
- **v3 (LLM-based):** Uses LLaMA-2 or Vicuna models with LoRA (Low-Rank Adaptation) adapters. The primary entry point involves loading the base model, applying the PEFT adapter, and using a specific prompt template: `Instruction: ... Input: ... Answer:`.

## 2. Environment Setup
- **Hardware:** Testing was conducted on an NVIDIA A10G (24GB VRAM). 8-bit quantization is necessary to fit the 7B/13B models within reasonable memory bounds.
- **Dependencies:** `transformers`, `peft`, and `bitsandbytes` are critical for running the LoRA-adapted Llama-2 models.

## 3. Findings
- **What Works:** The LoRA weights integrate seamlessly with the HuggingFace `peft` library. The zero-shot/few-shot sentiment performance on financial news is significantly higher than base LLMs.
- **Missing Documentation:** The original repo lacks a unified API wrapper. Most examples are provided as Jupyter notebooks, making it difficult to deploy as a microservice without refactoring.
- **Bugs/Issues:** Some model paths on HuggingFace refer to deprecated base model versions. We have updated our configuration to use `nousresearch/llama-2-7b-hf` as a stable base.

## 4. Architecture Rationale
A FastAPI wrapper was chosen for its asynchronous capabilities and native Pydantic support, allowing for high-performance I/O and strict input validation. The service utilizes a singleton pattern for the `SentimentAnalyzer` to ensure the model is only loaded once into memory.
