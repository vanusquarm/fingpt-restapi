# Friction Log: Development & Deployment

### 1. Model Size and Memory
- **Problem:** Llama-2-7b based FinGPT requires over 28GB of VRAM in full precision.
- **Solution:** Implemented 8-bit quantization using `bitsandbytes`. This reduced memory usage to ~10GB, allowing it to run on consumer-grade GPUs or smaller cloud instances.
- **Suggestion:** Provide a "Tiny" version of the API using FinBERT as an automated fallback for CPU-only environments.

### 2. Dependency Conflicts
- **Problem:** `bitsandbytes` often has issues with specific CUDA versions in Docker.
- **Solution:** Explicitly used the `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` base image to ensure driver compatibility.

### 3. Model Loading Time
- **Problem:** Loading 7B parameters + LoRA weights takes 1-2 minutes.
- **Solution:** Moved model loading to a background thread on startup and implemented a `/health` endpoint that returns a `loading` state until the model is ready.
- **Suggestion:** Use a persistent volume for HuggingFace cache in `docker-compose` to avoid redownloading 15GB+ on every container restart.

### 4. Prompt Engineering
- **Problem:** The model is sensitive to the prompt format. Missing the "Answer:" suffix causes it to generate rambling financial analysis instead of a single-word label.
- **Solution:** Hardcoded the instruction prompt in the service layer to ensure consistent output.
