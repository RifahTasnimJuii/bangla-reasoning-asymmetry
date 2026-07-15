# Reasoning Asymmetry in Bangla Mathematical NLP

Code and data for the paper **"Reasoning Asymmetry in Bangla Mathematical NLP: A Directional Evaluation of LLMs on GSM-Plus-BN."**

This repository contains the experiment pipeline, raw results, and summary statistics used to evaluate directional reasoning asymmetry (forward vs. reverse problem solving) in three LLaMA models on Bangla mathematical word problems.

## Overview

We evaluate three LLaMA models — LLaMA-3.1-8B-Instant, LLaMA-3.3-70B-Versatile, and LLaMA-4-Scout-17B — under three prompting strategies (zero-shot, few-shot, Chain-of-Thought) on the reversing-operation subset of the GSM-Plus-BN dataset. We introduce the **Reasoning Asymmetry Gap (RAG)** score — defined as forward accuracy minus reverse accuracy — and use McNemar's test to assess statistical significance across all nine model-condition pairs.

## Repository Contents

| File | Description |
|---|---|
| `run_experiment.py` | Main experiment script: loads sampled data, queries the Groq API for each model/condition, extracts numeric answers, computes accuracy, runs McNemar's test, and generates figures |
| `experiment_results.csv` | Full raw results — 9,000 evaluations (3 models × 3 prompting conditions × 500 forward/reverse pairs), including model outputs, extracted answers, and correctness flags |
| `summary_results.csv` | Aggregated forward accuracy, reverse accuracy, and RAG score for each of the 9 model-condition pairs |

## Dataset

This study uses the reversing-operation subset of the **GSM-Plus-BN** dataset, obtained via Mendeley Data: **[citation/DOI to be added]**.

Due to licensing, the original dataset is not redistributed in this repository. Please obtain it from the source above. The specific 500 sampled forward-reverse question pairs used in this study, along with all model outputs, are included in `experiment_results.csv`.

## Models

All models were accessed via the [Groq API](https://groq.com):
- `llama-3.1-8b-instant`
- `llama-3.3-70b-versatile`
- `meta-llama/llama-4-scout-17b-16e-instruct`

## Prompting Strategies

Prompt templates are defined in `run_experiment.py`:
- `zero_shot_prompt()` — question only, no examples
- `few_shot_prompt()` — question preceded by 3 fixed Bangla arithmetic examples
- `cot_prompt()` — instructs step-by-step Bangla reasoning before the final answer

**Answer extraction delimiter:** `####` — the model is instructed to place its final numeric answer immediately after this symbol. If the delimiter is not found, the last numeric value in the output is used as a fallback.

All prompting conditions use `temperature=0` and `max_tokens=512`.

## Setup

### Requirements

Install the required packages:

    pip install pandas numpy matplotlib seaborn statsmodels groq python-dotenv openpyxl

### API Key

Create a `.env` file in the project root (not included in this repo) with your Groq API key:

    GROQ_API_KEY=your_key_here

Multiple keys can be added as `GROQ_API_KEY_1`, `GROQ_API_KEY_2`, etc. for automatic rotation on rate limits.

### Running the experiment

    python run_experiment.py

This will:
1. Load and sample 500 forward-reverse pairs (`random_state=42`) from the reversing-operation subset
2. Query all 3 models under all 3 prompting conditions
3. Save results incrementally to `experiment_results.csv` (resumable if interrupted)
4. Compute accuracy, RAG scores, and McNemar's test results
5. Generate `rag_heatmap.png` and `forward_vs_reverse.png`

## Evaluation Metric — RAG Score

    RAG = Accuracy_forward − Accuracy_reverse

A higher RAG score indicates greater forward-direction bias. A score near zero can reflect either bidirectional competence or symmetric failure — the absolute accuracy values determine which interpretation applies.

## Citation

If you use this code or data, please cite:

[BibTeX entry to be added upon publication]

## License

MIT License
