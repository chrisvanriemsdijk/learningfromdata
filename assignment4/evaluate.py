import argparse
import numpy as np
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from helpers.helpers_general import read_corpus, remove_emojis, lemmatize, stem
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


def generate_tokens(lm, X):
    """Given the pretrained model, generate tokens"""

    # Set sequence length
    sequence_length = 150

    # Load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lm)

    # Tokenize sets
    tokens = tokenizer(X, padding=True, max_length=sequence_length, truncation=True, return_tensors="np").data
    return tokens


def evaluate(model, tokens, Y_bin):
    """Calculates accuracy given the model, tokens and true labels"""

    # Predict and get 'logits' (predicted label)
    Y_pred = model.predict(tokens)["logits"]

    # Transform to label
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_gold = np.argmax(Y_bin, axis=1)

    # Test and print performance
    print("Accuracy on own {1} set: {0}".format(round(accuracy_score(Y_gold, Y_pred), 3), "custom"))
    print("F1 score on own {1} set: {0}".format(round(f1_score(Y_gold, Y_pred), 3), "custom"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your program description here")

    parser.add_argument("--input_file", default="data/test.tsv", help="Input file to learn from")
    parser.add_argument("--output_file", default="data/results.txt", help="Output file")
    parser.add_argument("--model_name", default="model.pt", help="Model name stored in models/ dir")
    parser.add_argument("--lemmatize", action="store_true", help="Lemmatize text")
    parser.add_argument("--stem", action="store_true", help="Stem text")
    parser.add_argument("--emoji_remove", action="store_true", help="Remove emojis from text")

    args = parser.parse_args()

    X, Y = read_corpus(args.input_file)
    X = [[" ".join(subarray)] for subarray in X]

    if args.lemmatize:
        X = lemmatize(X)

    if args.stem:
        X = stem(X)

    if args.emoji_remove:
        X = remove_emojis(X)

    X = [text for sublist in X for text in sublist]
    # Transform string labels to one-hot encodings
    classes = ["OFF", "NOT"]
    Y_bin = np.zeros((len(Y), len(classes)))

    # Loop through the labels and set the corresponding class to 1
    for i, label in enumerate(Y):
        Y_bin[i, classes.index(label)] = 1

    num_labels = len(set(Y))

    model = TFAutoModelForSequenceClassification.from_pretrained("models", num_labels=num_labels)
    print(model)
    # Get tokens
    tokens = generate_tokens("distilbert-base-uncased", X)

    evaluate(model, tokens, Y_bin)
