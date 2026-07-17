# Sentiment classifier fine-tuning

This workspace contains a minimal Hugging Face fine-tuning pipeline for a binary sentiment classifier.

## Quick start

1. Create and activate a virtual environment:
   ```bash
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train_simple_sentiment.py --output-dir models/simple_sentiment
   ```
4. Run inference:
   ```bash
   python predict_simple.py --text "This movie was absolutely wonderful" --model-path models/simple_sentiment/model.pkl
   ```

The lightweight workflow uses a small handcrafted dataset so it can run reliably in a fresh environment.
