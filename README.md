# # LLM-Preprocessing for Noisy Sentiment Analysis

Using Large Language Models to clean noisy text before sentiment classification.

## Overview

Sentiment analysis models trained on clean text struggle with real-world noisy inputs (typos, slang, random capitalization). This project explores using LLMs (Claude Haiku and GPT-3.5-turbo) as a preprocessing step to clean noisy text before classification.

## Key Findings

| Condition | Accuracy | F1 Score |
|-----------|----------|----------|
| Clean Text | 92.5% | 92.3% |
| Noisy Text | 81.8% | 78.9% |
| Claude Cleaned | 92.4% | 92.3% |
| GPT Cleaned | 93.0% | 92.9% |

- **Noise causes 10.7% accuracy drop** in DistilBERT
- **LLM cleaning recovers 99-105%** of lost accuracy
- Noise creates **4x more false negatives**, systematically missing positive reviews
- Both LLMs restore balanced predictions

## Methodology

1. **Dataset:** IMDB movie reviews (1,000 test samples)
2. **Classifier:** DistilBERT fine-tuned on 20k clean reviews
3. **Noise Types:** Typos, elongation, keyboard errors, random caps, slang substitution
4. **Cleaning:** API-based LLM preprocessing with sentiment-preservation prompt

## Example

```
Original:  "This movie was really good"
Noisy:     "Tihs mvoie was fr fr fiiire"
Cleaned:   "This movie was really good"
```

## Tech Stack

- Python
- HuggingFace Transformers (DistilBERT)
- Anthropic API (Claude Haiku)
- OpenAI API (GPT-3.5-turbo)
- Google Colab (A100 GPU)

## Limitations

- Only tested on IMDB reviews (long-form text)
- Synthetic noise may differ from real social media noise
- Limited to two mid-tier LLMs
