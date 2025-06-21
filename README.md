# Deep Learning for NLP & Twitter Sentiment Analysis

A comprehensive repository containing two Jupyter notebooks that cover the complete journey from basic RNNs to state-of-the-art transformer models, plus practical sentiment analysis applications.

## 📚 What's Inside

This repository contains two main notebooks focused on Natural Language Processing using deep learning techniques.

### 1. Deep Learning for NLP - Zero to Transformers & BERT
**File:** `deep-learning-for-nlp-zero-to-transformers-bert.ipynb`

A complete educational journey covering the evolution of NLP architectures:

**🔧 Core Architectures Covered**
- Simple RNNs (Recurrent Neural Networks)
- Word Embeddings implementation and theory
- LSTM (Long Short-Term Memory) networks
- GRU (Gated Recurrent Units)
- Bi-Directional RNNs
- Encoder-Decoder Models (Seq2Seq)
- Attention Mechanisms
- Transformers Architecture ("Attention is all you need")
- BERT Implementation

**📊 Performance Results**
The notebook includes comprehensive model comparisons on toxic comment classification:
- SimpleRNN: 69.5% AUC
- LSTM: 96.0% AUC
- GRU: 97.2% AUC
- Bi-directional LSTM: 96.7% AUC
- BERT: State-of-the-art performance

**⚡ Technical Features**
- TPU configuration and optimization for large models
- GloVe embeddings integration (840B.300d)
- Multilingual BERT implementation using DistilBERT
- Production-ready code with proper data pipelines

### 2. Twitter Sentiment Analysis - EDA and Modeling
**File:** `twitter-sentiment-extaction-analysis-eda-and-model.ipynb`

A practical sentiment analysis project with comprehensive exploratory data analysis:

**📈 Dataset Overview**
- **Total samples:** 27,480 labeled tweets
- **Sentiment distribution:**
  - Neutral: 11,117 samples
  - Positive: 8,582 samples  
  - Negative: 7,781 samples

**🔍 Key Analysis Features**
- Text preprocessing and cleaning pipelines
- Jaccard score calculations for text similarity analysis
- Word frequency analysis by sentiment category
- Selected text extraction for explainable AI
- Comprehensive EDA with visualizations

**📝 Text Analysis Insights**
- Most common words across different sentiments
- Positive sentiment keywords: "good", "happy", "love", "thanks", "great"
- Negative sentiment keywords: "miss", "sad", "sorry", "bad", "hate"
- Neutral sentiment patterns and vocabulary analysis
