import os
import tempfile
import unittest

from train_simple_sentiment import build_samples, predict, train_model


class SentimentTrainingTests(unittest.TestCase):
    def test_build_samples_are_balanced(self):
        texts, labels = build_samples()
        self.assertGreater(len(texts), 20)
        self.assertEqual(labels.count(1), labels.count(0))

    def test_train_model_writes_model_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = train_model(tmpdir)
            self.assertIn("accuracy", metrics)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "model.pkl")))
            label, probability = predict("I loved this film and found it deeply moving.", os.path.join(tmpdir, "model.pkl"))
            self.assertEqual(label, "positive")
            self.assertGreaterEqual(probability, 0.0)


if __name__ == "__main__":
    unittest.main()
