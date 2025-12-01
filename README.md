# SarcClassifier

A complete machine learning pipeline for sarcasm detection in social media text, featuring both sarcasm detection and subtype classification capabilities.

## Overview

SarcClassifier is a two-stage NLP system that detects sarcasm in social media comments and classifies its type. The project provides an end-to-end solution from data preprocessing through model training to deployment as a FastAPI service.

## Main Components

### Data Processing
- **Raw Data**: Uses the Sarcasm Corpus v2 dataset containing labeled Reddit comments
- **Preprocessing Pipeline**: Text cleaning, normalization, and feature extraction using sentence transformers, using all-mpnet-base-v2 transformer model
- **Data Validation**: Schema validation using Pandera for data quality assurance

### Machine Learning Pipeline
- **Model Training**: XGBoost classifiers with Optuna hyperparameter optimization
- **Dual Classification**: Primary sarcasm detection + secondary subtype classification (GEN/HYP/RQ)
- **Experiment Tracking**: MLflow integration for model versioning and metrics tracking

### Inference Service
- **FastAPI Application**: RESTful API with health checks and prediction endpoints
- **Real-time Processing**: Live text preprocessing and prediction
- **Dual Predictions**: Both sarcasm detection and subtype classification available

## Installation

1. Clone the repository:
```bash
git clone https://github.com/EsotericMat/SarcClassifierE2E.git
cd SarcClassifierE2E
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Pipeline

Run the complete pipeline:
```bash
python main.py
```

Run specific steps:
```bash
# Data preprocessing only
python main.py --step preprocess

# Training only
python main.py --step train --target sarcasm
```

### Inference Service

Start the API server:
```bash
python3 -m inference.app
```

The service provides:
- Health check: `GET /health`
- Sarcasm detection: `POST /predict_sarc`
- Subtype classification: `POST /predict_sarc_subclass`

