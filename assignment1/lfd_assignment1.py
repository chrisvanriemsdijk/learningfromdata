#!/usr/bin/env python

"""TODO: add high-level description of this Python script"""


# Apply the default theme
import argparse
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tf",
        "--train_file",
        default="train.txt",
        type=str,
        help="Train file to learn from (default train.txt)",
    )
    parser.add_argument(
        "-df",
        "--dev_file",
        default="dev.txt",
        type=str,
        help="Dev file to evaluate on (default dev.txt)",
    )
    parser.add_argument(
        "-s",
        "--sentiment",
        action="store_true",
        help="Do sentiment analysis (2-class problem)",
    )
    parser.add_argument(
        "-t",
        "--tfidf",
        action="store_true",
        help="Use the TF-IDF vectorizer instead of CountVectorizer",
    )
    parser.add_argument(
        "-nb",
        "--naive_bayes",
        action="store_true",
        help="Use the NB pipeline",
    )
    parser.add_argument(
        "-knn",
        "--k_nearest_neighbour",
        action="store_true",
        help="Use the KNN pipeline",
    )
    parser.add_argument(
        "-SVM",
        "--support_vector_machine",
        action="store_true",
        help="Use the SVM pipeline",
    )
    args = parser.parse_args()
    return args


def read_corpus(corpus_file, use_sentiment):
    """TODO: add function description"""
    documents = []
    labels = []
    with open(corpus_file, encoding="utf-8") as in_file:
        for line in in_file:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])
    return documents, labels


def identity(inp):
    """Dummy function that just returns the input"""
    return inp


def check_balance(y):
    data = {}
    for label in y:
        if not data.get(label):
            data[label] = 1
        data[label] += 1
    labels = data.keys()
    values = data.values()
    plt.bar(labels, values)
    print(data)
    plt.show()


if __name__ == "__main__":
    args = create_arg_parser()

    # TODO: comment
    X_train, Y_train = read_corpus(args.train_file, args.sentiment)
    X_test, Y_test = read_corpus(args.dev_file, args.sentiment)

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if args.tfidf:
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    else:
        # Bag of Words vectorizer
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity)

    classifiers = []
    if args.naive_bayes:
        classifiers = [
            ("Multinomial NB", MultinomialNB()),
        ]
    if args.k_nearest_neighbour:
        classifiers = []
    if args.support_vector_machine:
        classifiers = []

    # Combine the vectorizer with a Naive Bayes classifier
    # Of course you have to experiment with different classifiers
    # You can all find them through the sklearn library
    for name, classifier in classifiers:
        pipeline = Pipeline([("vec", vec), ("cls", classifier)])

        # TODO: comment this
        pipeline.fit(X_train, Y_train)

        # TODO: comment this
        Y_pred = pipeline.predict(X_test)

        # TODO: comment this
        acc = accuracy_score(Y_test, Y_pred)
        cm = confusion_matrix(Y_test, Y_pred)
        print(classification_report(Y_test, Y_pred))
        print(f"Final accuracy: {acc}")
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=classifier.classes_
        )
        disp.plot()
        disp.ax_.set_title(name)
        plt.show()
