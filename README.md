# Shrine Seeker Compendium: Apple MLX LoRA Fine-Tuning Pipeline

This repository contains the dataset, architecture, and execution pipeline for fine-tuning state-of-the-art Large Language Models (LLMs) natively on Apple Silicon using the MLX framework. The primary objective is to inject highly specific, granular domain knowledge (The Legend of Zelda: Breath of the Wild Hyrule Compendium) into a foundation model's weights, creating an authoritative, hallucination-free lore assistant.

## Phase 1: The Foundation Model Architecture

Instead of pre-training from scratch, this pipeline leverages existing multi-billion parameter foundation models (e.g., Meta's Llama 3 8B or Llama 4 Maverick). 

Foundation models are dense neural networks that have already processed trillions of tokens of internet data. Because of this massive pre-training phase, the model already possesses deep structural knowledge of the English language, logical reasoning capabilities, and general world knowledge. 

**Target Architecture (Mac Studio / 512GB Unified Memory):**
The production target for this pipeline is **Llama 4 Maverick**. Maverick utilizes a **Mixture-of-Experts (MoE)** architecture. Instead of pushing data through all 400 billion parameters for every single word, it features a routing network that selects only the most relevant "expert" sub-networks (activating only 17 billion parameters per token). This allows for staggering intelligence and deep reasoning while maintaining incredibly fast generation speeds.

## Phase 2: The Fine-Tuning Methodology (LoRA)

To teach the foundation model the highly specific facts of the Hyrule Compendium, we use a technique called **LoRA (Low-Rank Adaptation)**.

### How LoRA Works:
1. **Freezing the Brain:** We completely "freeze" the original 400 billion weights of the foundation model. We do not alter its core understanding of grammar or general logic.
2. **Injecting Adapters:** We inject tiny, trainable matrices (adapters) into the model's attention layers. 
3. **Targeted Training:** During the training loop, only these tiny adapter matrices are updated. 

**The Result:** Instead of requiring clusters of enterprise GPUs to update billions of weights, LoRA allows us to train a highly specialized ~10-million parameter adapter natively on a single Apple Silicon chip. During inference, these adapters act as a "lens" that slightly bends the foundation model's logic toward our specific Zelda data.

## Phase 3: Dataset Architecture

The dataset (`data/train.jsonl`) is formatted as JSON Lines. Because the base model has already undergone "Instruction Tuning," it expects data to be formatted as a conversation rather than raw text. 

We wrap our Compendium data in specific structural tokens (`<|user|>` and `<|assistant|>`) so the model learns exactly when to listen and when to output facts.

**Example Entry:**
```json
{"text": "<|user|>\nWhat is a Silver Lynel, and what are its characteristics?\n<|assistant|>\nSilver Lynels are not to be trifled with. They have been influenced by Ganon's fiendish magic, so they are the strongest among the Lynel species... Upon defeat or collection, it can yield: lynel horn, lynel hoof, lynel guts, topaz, ruby, sapphire, diamond, star fragment."}
```

## Phase 4: Execution & Deployment

This pipeline utilizes Apple's `mlx-lm` library to natively compile the tensor operations for the Metal Performance Shaders (MPS) backend, utilizing unified memory to eliminate data transfer bottlenecks.

### 1. Execute the LoRA Training Loop
Run this command from the root directory to initiate the fine-tuning process. The framework will automatically fetch the base model, apply 4-bit quantization, inject the LoRA adapters, and begin training on `data/train.jsonl`.

```bash
python3 -m mlx_lm.lora \
    --model mlx-community/Meta-Llama-3-8B-Instruct-4bit \
    --data data \
    --train \
    --iters 2000 \
    --batch-size 2
```
*(Note: Swap the `--model` flag to the Llama 4 Maverick repository ID when deploying on the Mac Studio).*

### 2. Validation & Inference Testing
Once training is complete, the custom weights are saved in the `adapters` directory. Use the generation CLI to merge the base model with your custom adapters on the fly and test the model's new domain knowledge.

```bash
python3 -m mlx_lm.generate \
    --model mlx-community/Meta-Llama-3-8B-Instruct-4bit \
    --adapter-path adapters \
    --max-tokens 200 \
    --prompt "<|user|>\nWhat is the material Hearty Durian used for, and where is it found?\n<|assistant|>\n"
```