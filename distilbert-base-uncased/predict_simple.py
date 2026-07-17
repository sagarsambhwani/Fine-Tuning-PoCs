import argparse
from train_simple_sentiment import predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment for a given text")
    parser.add_argument("--text", required=True)
    parser.add_argument("--model-path", default="models/simple_sentiment/model.pkl")
    args = parser.parse_args()
    label, probability = predict(args.text, args.model_path)
    print({"label": label, "positive_probability": round(probability, 4)})
