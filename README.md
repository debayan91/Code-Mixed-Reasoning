# Code-Mixed Mathematical Reasoning via Synthetic Chain-of-Thought

**Status:** Work in Progress (Phase 1 Complete, Phase 2 Ongoing)
**Base Model:** Llama-3-8B-Instruct
**Framework:** Unsloth / QLoRA

## Project Abstract

This project investigates the "Translation Barrier" hypothesis in multilingual Large Language Models (LLMs). Current models typically process code-mixed (Hinglish) queries by implicitly translating them into English for internal reasoning, which introduces latency and semantic drift.

We propose a method to align the model's "Language of Thought" with the input language. By fine-tuning a Small Language Model (SLM) on a synthetically distilled dataset of Romanized Hinglish Chain-of-Thought (CoT) reasoning traces, we aim to improve logical accuracy and pedagogical clarity in low-resource settings.

## Objectives

1.  **Synthetic Distillation:** Create a dataset of mathematical reasoning problems where the intermediate logical steps are performed in Romanized Hinglish, adhering to the Matrix Language Frame (MLF) theory.
2.  **Low-Resource Fine-Tuning:** Demonstrate that complex reasoning capabilities can be instilled in an 8-parameter model (Llama-3-8B) using Quantized Low-Rank Adaptation (QLoRA) on consumer-grade hardware (Single T4 GPU).
3.  **Ablation Study:** Compare the performance of Hinglish-CoT against English-CoT and Zero-Shot baselines to quantify the benefits of language alignment.

## Methodology

The project follows a three-phase execution protocol:

### Phase 1: Dataset Generation (Completed)
We constructed **Hinglish-GSM8K**, a dataset derived from the standard GSM8K benchmark. Using a Teacher-Student distillation approach (Teacher: Gemini 1.5 Pro), we synthesized reasoning traces that enforce:
* **Matrix Language (Hindi):** Grammatical structure, verbs, and connectives.
* **Embedded Language (English):** Mathematical entities, variables, numbers, and technical nouns.

The dataset is categorized into three difficulty tiers to facilitate granular analysis:
1.  **Lexical/Arithmetic:** Single-step operations to test linguistic coherence.
2.  **Story Problems:** Standard GSM8K-level word problems requiring context retention.
3.  **Advanced Logic:** Multi-variable problems requiring distinct planning steps.

### Phase 2: Training (Current)
We employ Parameter-Efficient Fine-Tuning (PEFT) using the Unsloth library.
* **Technique:** QLoRA (4-bit NormalFloat Quantization).
* **Target Modules:** Attention mechanisms (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and Feed-Forward Networks (`gate_proj`, `up_proj`, `down_proj`).
* **Hardware Constraint:** Optimization for <16GB VRAM (NVIDIA T4).

### Phase 3: Evaluation (Planned)
The model will be evaluated using Exact Match (EM) accuracy on the final numerical output. We will conduct a micro-analysis of failure modes, specifically distinguishing between logic errors and language hallucination errors.

## Dataset Structure

The training data follows the Alpaca-style JSONL format.

**Example Entry:**
```json
{
  "instruction": "Solve the following math problem in Hinglish explicitly showing your steps.",
  "input": "If cost price is $100 and profit is 20%, what is selling price?",
  "output": "Cost Price (CP) $100 hai. Profit percentage 20% diya gaya hai. Selling Price (SP) nikalne ke liye hum formula use karenge: SP = CP + Profit. Pehle profit value nikalte hain: 20% of 100 = $20. Ab SP = 100 + 20 = 120. #### 120"
}
