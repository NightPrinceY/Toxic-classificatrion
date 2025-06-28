# Toxic-Predict

Toxic-Predict is a machine learning project developed as part of the Cellula Internship, focused on safe and responsible multi-modal toxic content moderation. It classifies text queries and image descriptions into nine toxicity categories such as "Safe", "Violent Crimes", "Non-Violent Crimes", "Unsafe", and others. The project leverages deep learning (Keras/TensorFlow), NLP preprocessing, and benchmarking with modern transformer models to build and evaluate a robust multi-class toxic content classifier.

---

## 🚩 Project Context

This project is part of the **Cellula Internship** proposal:  
**"Safe and Responsible Multi-Modal Toxic Content Moderation"**  
The goal is to build a dual-stage moderation pipeline for both text and images, combining hard guardrails (Llama Guard) and soft classification (DistilBERT/Deep Learning) for nuanced, policy-compliant moderation.

---

## Project Structure

```
.
├── app.py
├── run.py
├── test.py
├── requirements.txt
├── README.md
├── data/
│   ├── cellula-toxic.csv
│   ├── cleaned.csv
│   ├── eval.csv
│   ├── test.csv
│   ├── tokenizer.pkl
│   └── train.csv
├── models/
│   ├── model.py
│   ├── toxic_classifier.h5
│   └── toxic_classifier.keras
├── notebooks/
│   ├── Preprocessing.ipynb
│   └── tokenization.ipynb
└── src/
    ├── preprocess.py
    └── tokenize_and_split.py
```

---

## Features

- Dual-stage moderation: hard filter (Llama Guard) + soft classifier (DistilBERT/CNN/LSTM)
- Data cleaning, preprocessing, and label encoding
- Tokenization and sequence padding for text data
- Deep learning and transformer-based models for multi-class toxicity classification
- Evaluation metrics: classification report and confusion matrix
- Jupyter notebooks for data exploration and model development
- Streamlit web app for demo and deployment

---

## Setup

1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/toxic-classification.git
   cd toxic-predict
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Prepare data**
   - Place your data files in the `data/` directory if not already present.

4. **Train the model**
   - Use the scripts in `src/` or the Jupyter notebooks in `notebooks/` to preprocess data and train the model.

5. **Run predictions**
   - Use `app.py` or `run.py` to run inference on new data.

---

## Usage

- **Preprocessing and Tokenization:**  
  See `notebooks/Preprocessing.ipynb` and `notebooks/tokenization.ipynb` for step-by-step data cleaning, splitting, and tokenization.
- **Model Training:**  
  Model architecture and training code are in `models/model.py`.
- **Inference:**  
  Load the trained model (`models/toxic_classifier.h5` or `.keras`) and tokenizer (`data/tokenizer.pkl`) to predict toxicity categories for new samples.

---

## Data

- CSV files with columns: `query`, `image descriptions`, `Toxic Category`, and `Toxic Category Encoded`.
- Data splits: `train.csv`, `eval.csv`, `test.csv`, and `cleaned.csv` for processed data.
- 9 categories: Safe, Violent Crimes, Elections, Sex-Related Crimes, Unsafe, Non-Violent Crimes, Child Sexual Exploitation, Unknown S-Type, Suicide & Self-Harm.

---

## Model

- Deep learning model built with Keras (TensorFlow backend).
- Multi-class classification with label encoding for toxicity categories.
- Benchmarking with PEFT-LoRA DistilBERT and baseline CNN/LSTM.

---

## Evaluation

- Classification report and confusion matrix are generated for model evaluation.
- See the evaluation steps in `notebooks/Preprocessing.ipynb`.

---

## Hugging Face Model Card

```yaml
---
language: en
tags:
  - toxic-content
  - text-classification
  - keras
  - deep-learning
license: mit
datasets:
  - custom
metrics:
  - accuracy
  - f1
model-index:
  - name: Toxic-Predict
    results: []
---
# Toxic-Predict

A deep learning model for multi-class toxic content classification. See [GitHub](https://github.com/yourusername/toxic-predict) for code and training details.
```

---

## 📚 Resources

- [Cellula Internship Project Proposal](#)  
- [BLIP: Bootstrapped Language-Image Pre-training](https://github.com/salesforce/BLIP)
- [Llama Guard](https://llama.meta.com/llama-guard/)
- [DistilBERT](https://huggingface.co/distilbert-base-uncased)
- [Streamlit](https://streamlit.io/)

---

## License

MIT License

---

**Author:** Yahya Muhammad Alnwsany  
**Contact:** yahyaalnwsany39@gmail.com  
**Portfolio:** https://nightprincey.github.io/Portfolio/
