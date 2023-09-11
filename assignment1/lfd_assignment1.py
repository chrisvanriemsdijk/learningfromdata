#!/usr/bin/env python

"""TODO: add high-level description of this Python script"""


# Apply the default theme
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


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


def get_scores(key: str, report_dict):
    dict_values = report_dict[key]
    return dict_values.values()


def save_confusion_matrix(Y_test, Y_pred, classifier, name):
    cm = confusion_matrix(Y_test, Y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=classifier.classes_
    )
    disp.plot()
    disp.ax_.set_title(name)
    disp.figure_.savefig(f"cm/{name}.png", dpi=300)


def setup_df():
    return pd.DataFrame(
        {
            "classifier": [],
            "hyperparameters": [],
            "vectorizer": [],
            "accuracy": [],
            "books_p": [],
            "books_r": [],
            "books_f1": [],
            "camera_p": [],
            "camera_r": [],
            "camera_f1": [],
            "dvd_p": [],
            "dvd_r": [],
            "dvd_f1": [],
            "health_p": [],
            "health_r": [],
            "health_f1": [],
            "music_p": [],
            "music_r": [],
            "music_f1": [],
            "software_p": [],
            "software_r": [],
            "software_f1": [],
        }
    )


def run_experiments(classifiers, vec, X_train, Y_train, X_test, Y_test):
    results = []
    for name, (classifier, hyperparameters) in classifiers:
        pipeline = Pipeline([("vec", vec), ("cls", classifier)])

        # TODO: comment this
        pipeline.fit(X_train, Y_train)

        # TODO: comment this
        Y_pred = pipeline.predict(X_test)

        # TODO: comment this
        acc = accuracy_score(Y_test, Y_pred)
        report_dict = classification_report(Y_test, Y_pred, output_dict=True)
        save_confusion_matrix(Y_test, Y_pred, classifier, name)
        b_p, b_r, b_f1, _ = get_scores("books", report_dict)
        c_p, c_r, c_f1, _ = get_scores("camera", report_dict)
        d_p, d_r, d_f1, _ = get_scores("dvd", report_dict)
        h_p, h_r, h_f1, _ = get_scores("health", report_dict)
        m_p, m_r, m_f1, _ = get_scores("music", report_dict)
        s_p, s_r, s_f1, _ = get_scores("software", report_dict)

        # TODO ADD vectorizer name
        results.append(
            [
                name,
                hyperparameters,
                vec.__class__,
                acc,
                b_p,
                b_r,
                b_f1,
                c_p,
                c_r,
                c_f1,
                d_p,
                d_r,
                d_f1,
                h_p,
                h_r,
                h_f1,
                m_p,
                m_r,
                m_f1,
                s_p,
                s_r,
                s_f1,
            ]
        )
    return results


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
        name = "nb"
        classifiers = [
            ("Multinomial NB", (MultinomialNB(), "Nothing")),
        ]
    if args.k_nearest_neighbour:
        name = "knn"
        classifiers = []
    if args.support_vector_machine:
        name = "svm"
        classifiers = []

    # Combine the vectorizer with a Naive Bayes classifier
    # Of course you have to experiment with different classifiers
    # You can all find them through the sklearn library
    df = setup_df()
    results = run_experiments(classifiers, vec, X_train, Y_train, X_test, Y_test)
    df_extended = pd.DataFrame(results, columns=df.columns)
    df = pd.concat([df, df_extended])
    df.to_excel(f"results/{name}.xlsx")
