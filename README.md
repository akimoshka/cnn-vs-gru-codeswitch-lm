# CNN vs RNN Language Models for Russian-English Code-Switching Streams

This repository contains an NLP case study for the **NLP 2026 course**.

The goal of the project is to compare two neural language modeling architectures:

- **GRU Language Model (RNN-based)**
- **Causal CNN Language Model**

Both models were trained on a synthetic **Russian-English code-switching dataset** built from bilingual parallel corpora.

The main research question:

> Which architecture better models mixed-language text, predicts next tokens, and handles switch points between Russian and English?

---

# Motivation

Code-switching is common in multilingual communication, especially online:

- “Я just finished работу”
- “Мы had meeting вчера”

Such text combines multiple languages inside one sentence.

This creates challenges for language models:

- language switching
- mixed grammar
- long-range dependencies
- token distribution shifts

This project studies whether:

- **GRU** models sequential dependencies better
- **CNN** models local bilingual patterns better

---

# Project Structure

```text
project/
│── data/
│   ├── train.json
│   ├── val.json
│   └── test.json
│
│── notebooks/
│   ├── 01_data_and_eda.ipynb
│   ├── 02_train_gru_lm.ipynb
│   ├── 03_train_cnn_lm.ipynb
│   └── 04_analysis_and_plots.ipynb
│
│── experiments/
│   └── saved_models/
│
│── results/
│   ├── tables/
│   └── figures/
│
└── README.md
````

---

# Dataset Creation

The dataset was created using bilingual sentence pairs from the multilingual corpus OPUS-100.

## Pipeline

1. Load English-Russian parallel sentences
2. Randomly mix English and Russian fragments
3. Generate synthetic code-switched sentences
4. Detect token language:

   * RU = Cyrillic token
   * EN = Latin token
5. Create switch-point labels

Example:

```text
Original:
EN: I need to go home
RU: Мне нужно идти домой

Mixed:
I need идти домой
```

---

# Models

## 1. GRU Language Model

Architecture:

* Embedding layer
* 1-layer GRU
* Linear output layer

Good for:

* sequential memory
* context tracking
* long-span dependencies

---

## 2. CNN Language Model

Architecture:

* Embedding layer
* Causal 1D convolutions
* Dilated convolution blocks
* Linear output layer

Good for:

* parallel computation
* local patterns
* efficient training

---

# Experimental Setup

To keep experiments reproducible on laptop hardware:

* Train subset: 10,000 samples
* Validation subset: 2,000 samples
* Test subset: 2,000 samples

Shared settings:

* Embedding size: 64
* Hidden size: 128
* Batch size: 16
* Epochs: 3
* Max sequence length: 20
* Adam optimizer
* Learning rate: 0.001

---

# Evaluation Metrics

## Language Modeling

* Cross-Entropy Loss
* Perplexity

## Switch Prediction

* Switch Accuracy

Switch point = whether language changes between adjacent tokens.

Example:

```text
Я люблю coding today

RU RU EN EN
0  1  0  0
```

---

# Final Results

| Model  | Test Loss | Test Perplexity | Switch Accuracy |
| ------ | --------: | --------------: | --------------: |
| GRU LM |     4.031 |           56.32 |           0.748 |
| CNN LM |     4.237 |           69.23 |           0.748 |

---

# Main Findings

## GRU outperformed CNN for language modeling

Lower perplexity means better next-token prediction.

GRU:

```text
56.32
```

CNN:

```text
69.23
```

This suggests recurrent memory helped model mixed-language sequences.

---

## Switch Accuracy was similar

Both models reached:

```text
~0.748
```

This likely means switch prediction is imbalanced, since many token transitions are non-switches.

Accuracy alone is not enough.

Future work should add:

* Precision
* Recall
* F1-score for switch class

---

# Visualizations

Generated figures are stored in:

```text
results/figures/
```

Includes:

* Perplexity comparison
* Validation learning curves
* Loss curves
* Radar chart
* Metric summary tables

---

# How to Run

## Install dependencies

```bash
pip install torch pandas numpy matplotlib tqdm scikit-learn
pip install datasets
```

---

## Run notebooks in order

```text
01_data_and_eda.ipynb
02_train_gru_lm.ipynb
03_train_cnn_lm.ipynb
04_analysis_and_plots.ipynb
```

---

# Limitations

* Synthetic code-switching data
* Small training subset
* Only 3 training epochs
* Simple switch metric

---

# Future Improvements

* Real Russian-English social media corpus
* Transformer baseline
* F1-score for switch prediction
* Human evaluation of generated text
* Larger training setup

