import argparse
from typing import List
import numpy as np
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from helpers.helpers_general import read_corpus, remove_emojis, lemmatize, stem
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt


def generate_tokens(lm: str, X: List[str]):
    """Given the pretrained tokenizer, generate tokens"""

    # Set sequence length
    sequence_length = 150

    # Load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lm)

    # Tokenize sets
    tokens = tokenizer(X, padding=True, max_length=sequence_length, truncation=True, return_tensors="np").data
    return tokens


def plot_confusion_matrix(conf_matrix, classes: List[str], result_dir: str, name: str):
    """
    Takes a confusion matrix and class names, computes confusion matrix
    :param conf_matrix: Confusion matrix for dataset
    :type conf_matrix: sklearn.ConfusionMatrix
    :param classes: Classes available for confusion matrix (binary in this case)
    :param result_dir: Directory to save confusion matrix to
    :param name: Name of confusion matrix image
    """

    # Configure
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.get_cmap("GnBu"))
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Create plot
    fmt = "d"
    thresh = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(
                j,
                i,
                format(conf_matrix[i, j], fmt),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black",
            )

    # Set labels
    plt.ylabel("Gold label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"{result_dir}/confusion_{name}.png", dpi=300)


def evaluate(model, tokens, Y_bin, X, output_file):
    """Calculates accuracy given the model, tokens and true labels"""

    # Ptedict and get 'logits' (predicted label)
    Y_pred = model.predict(tokens)["logits"]

    # Transform to label
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_gold = np.argmax(Y_bin, axis=1)

    # Set class names and compute confusion matrix
    class_names = ["OFF", "NOT"]
    conf_matrix = confusion_matrix(Y_gold, Y_pred)

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, class_names, "data", "evaluate")
    plt.show()

    with open(output_file, "w") as f:
        # Test and print performance
        acc = round(accuracy_score(Y_gold, Y_pred), 3)
        print("Accuracy on own {1} set: {0}".format(acc, "custom"))
        f.write("Accuracy on own {1} set: {0}\n".format(acc, "custom"))

        # F1 score
        f1 = f1_score(Y_gold, Y_pred, average=None)
        print("\nF1 score on own {0} set: {1}".format("custom", f1))
        f.write("\nF1 score on own {0} set: {1}\n".format("custom", f1))

        # Print wrong (10) predicted reviews
        cnt = 0
        print("\nWrong predicted on own {0} set:".format("custom"))
        f.write("\nWrong predicted on own {0} set:\n".format("custom"))
        print("gold | pred | review")
        f.write("\ngold | pred | review\n")
        for gold, pred, text in zip(Y_gold, Y_pred, X):
            if gold != pred:
                cnt += 1
                print(class_names[gold], class_names[pred], text)
                f.write(f"{class_names[gold]}, {class_names[pred]}, {text}\n")
                if cnt == 10:
                    break

        # Print correct (10) predicted reviews
        cnt = 0
        print("\nCorrect predicted on own {0} set:".format("custom"))
        f.write("\nCorrect predicted on own {0} set:".format("custom"))
        print("gold | pred | review")
        f.write("gold | pred | review")
        for gold, pred, text in zip(Y_gold, Y_pred, X):
            if gold == pred:
                cnt += 1
                print(class_names[gold], class_names[pred], text)
                f.write(f"{class_names[gold]}, {class_names[pred]}, {text}\n")
                if cnt == 10:
                    break


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
    # Get tokens
    tokens = generate_tokens("distilbert-base-uncased", X)

    evaluate(model, tokens, Y_bin, X, args.output_file)
