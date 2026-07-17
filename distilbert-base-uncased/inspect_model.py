import pickle

from train_simple_sentiment import build_samples

texts, labels = build_samples()
print('samples', len(texts), 'positives', sum(labels), 'negatives', len(labels) - sum(labels))

with open('models/simple_sentiment/model.pkl', 'rb') as fh:
    vectorizer, model = pickle.load(fh)

for phrase in [
    'This movie was absolutely wonderful',
    'I loved this film and found it deeply moving',
    'The movie was boring and terrible',
    'It was a great experience and I enjoyed it',
]:
    vec = vectorizer.transform([phrase])
    print(phrase, model.predict(vec)[0], model.predict_proba(vec)[0])
