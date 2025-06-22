# Disaster Tweet Classification Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

## 🚀 Overview

A high-recall text classification pipeline for detecting disaster-related tweets. This repository delivers an end-to-end solution combining:

* **Bag-of-Words (BoW)** for sparse lexical features
* **Word2Vec** embeddings to capture domain-specific semantics
* A **fine-tuned bi-directional LSTM** for sequence modeling
* A **BoW + Logistic Regression** ensemble
* **Threshold tuning** to maximize recall

Achieved a private leaderboard recall of **0.97727**.

## 📈 Key Features

* **Preprocessing**: Cleans tweets by removing URLs, mentions, and non-letter characters, and normalizing whitespace.
* **Custom Word2Vec**: Learns embeddings on the disaster-tweet corpus for improved semantic representation.
* **Fine-tuned Embeddings**: Embedding layer is trainable for end-to-end optimization.
* **Class Weighting**: Addresses class imbalance by penalizing false negatives more heavily.
* **Early Stopping**: Restores best validation weights to prevent overfitting.
* **Ensemble Strategy**: Averages predictions of RNN and BoW-LogReg models.
* **Threshold Search**: Identifies optimal decision threshold on validation data to maximize recall.

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📂 Repository Structure

```
.
├── data/
│   ├── train.csv        # Training data
│   └── test_x.csv       # Test data for submission
├── notebooks/
│   └── rnn-text-class.ipynb  # Main analysis and model pipeline
├── .gitignore           # Ignored files and folders
├── .gitattributes       # Git attributes
├── LICENSE              # MIT license
├── README.md            # Project overview (this file)
└── requirements.txt     # Python dependencies
```

## ⚙️ Usage

1. Update data paths in the notebook if needed (default points to `/kaggle/input/txtclas-rnn-train/`).
2. Open `notebooks/rnn-text-class.ipynb` in Jupyter or Kaggle and run all cells.
3. Export `submission.csv` and submit to the competition to evaluate recall.

## 📊 Results

* **Validation Recall**: Tuned via threshold search on validation split.
* **Private Leaderboard Recall**: **0.97727**.

## ✨ Next Steps

* Experiment with pre-trained transformers (e.g., BERT, RoBERTa).
* Add convolutional layers or deeper RNN stacks.
* Incorporate tweet metadata (e.g., geolocation, user details, timestamps).
* Explore data augmentation for low-resource scenarios.

## 📝 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## 🙏 Acknowledgements

* **Kaggle**: Natural Language Processing with Disaster Tweets competition
* **TensorFlow**, **scikit-learn**, **Gensim**: Core libraries
* Inspired by best practices in text classification and deep learning.
