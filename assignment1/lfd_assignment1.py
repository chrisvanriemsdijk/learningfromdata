#!/usr/bin/env python

"""TODO: add high-level description of this Python script"""


# Apply the default theme
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from empath import Empath
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

nlp = spacy.load("en_core_web_sm")


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
        "-test",
        "--test_file",
        default="test.txt",
        type=str,
        help="Test file to test on (default test.txt)",
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
        "-rs",
        "--rangestart",
        default=1,
        type=int,
        help="Start of the n-gram range",
    )
    parser.add_argument(
        "-re",
        "--rangeend",
        default=1,
        type=int,
        help="End of the n-gram range",
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
        "-svm",
        "--support_vector_machine",
        action="store_true",
        help="Use the SVM pipeline",
    )
    parser.add_argument(
        "-dt",
        "--decision_trees",
        action="store_true",
        help="Use the DT pipeline",
    )
    parser.add_argument(
        "-rf", "--random_forest", action="store_true", help="Use the RF pipeline"
    )
    parser.add_argument(
        "-ens", "--ensemble", action="store_true", help="Use ensemble method"
    )
    parser.add_argument(
        "-wc", "--word_count", action="store_true", help="Use word count as a feature"
    )
    parser.add_argument(
        "-c",
        "--correlated",
        action="store_true",
        help="Remove highly correlated features",
    )
    parser.add_argument(
        "-dir",
        "--result_dir",
        default="results",
        type=str,
        help="Folder to store results to. (Default: results)",
    )
    parser.add_argument(
        "-ctf",
        "--count_tf_idf",
        action="store_true",
        help="Combine bag of words and TF-IDF vectorizers",
    )
    parser.add_argument(
        "-pos",
        "--part_of_speech",
        action="store_true",
        help="Add POS features to count vectorizer",
    )
    parser.add_argument(
        "-l", "--lemmatize", action="store_true", help="Lemmatize input"
    )
    parser.add_argument(
        "-pos_tf",
        "--part_of_speech_tf_idf",
        action="store_true",
        help="Combined features with BOW, POS and TF-IDF",
    )
    parser.add_argument("-st", "--stem", action="store_true", help="Store input")
    parser.add_argument("-b", "--best", action="store_true", help="Run the best model")
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


def identity_string(inp):
    return " ".join(inp)


class RemoveCorrelated(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, list):
            X = X.todense()
        X = pd.DataFrame(X)

        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
        print(to_drop)
        for i in to_drop:
            print(X[i])
            X[i] = 0
            print(X[i])

        print(type(X))
        print(X)
        X = csr_matrix(X.values)
        return X


def spacy_pos(txt):
    return [token.pos_ for token in nlp(txt)]


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
    if not os.path.exists("cm"):
        os.makedirs("cm")
    disp.figure_.savefig(f"cm/{name}.png", dpi=300)
    print(f"Confusion matrix of classifier {name} is saved to cm/{name}.png")


def add_word_count(x):
    lexicon = Empath()
    n = []
    for i in x:
        add = lexicon.analyze(i, normalize=True).values()
        add = list(add)
        add = np.array(add)
        n.append(add)
    return n


def lemmatize(x):
    lemmatizer = WordNetLemmatizer()
    new_docs = []
    for doc in x:
        new_docs.append([lemmatizer.lemmatize(word) for word in doc])
    return new_docs


def stem(x):
    stemmer = SnowballStemmer("english")
    new_docs = []
    for doc in x:
        new_docs.append([stemmer.stem(word) for word in doc])
    return new_docs


def setup_df():
    return pd.DataFrame(
        {
            "classifier": [],
            "vectorizer": [],
            "accuracy": [],
            "macro_precision": [],
            "macro_recall": [],
            "macro_f1": [],
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


def run_experiments(classifiers, vec, vec_name, X_train, Y_train, X_test, Y_test):
    # TODO Add function description
    results = []
    for name, classifier in classifiers:
        pipe = []

        if not args.word_count:
            pipe.append(("vec", vec))

        if args.correlated:
            pipe.append(("corr", RemoveCorrelated()))

        pipe.append(("cls", classifier))
        pipeline = Pipeline(pipe)

        # TODO: comment this
        pipeline.fit(X_train, Y_train)

        # TODO: comment this
        Y_pred = pipeline.predict(X_test)


        # TODO: comment this
        acc = accuracy_score(Y_test, Y_pred)
        macro_precision = precision_score(Y_test, Y_pred, average="macro")
        macro_recall = recall_score(Y_test, Y_pred, average="macro")
        macro_f1 = f1_score(Y_test, Y_pred, average="macro")
        report_dict = classification_report(Y_test, Y_pred, output_dict=True)
        print(classification_report(Y_test, Y_pred))
        save_confusion_matrix(Y_test, Y_pred, classifier, name)
        b_p, b_r, b_f1, _ = get_scores("books", report_dict)
        c_p, c_r, c_f1, _ = get_scores("camera", report_dict)
        d_p, d_r, d_f1, _ = get_scores("dvd", report_dict)
        h_p, h_r, h_f1, _ = get_scores("health", report_dict)
        m_p, m_r, m_f1, _ = get_scores("music", report_dict)
        s_p, s_r, s_f1, _ = get_scores("software", report_dict)

        results.append(
            [
                name,
                vec_name,
                acc,
                macro_precision,
                macro_recall,
                macro_f1,
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
    if args.test_file:
        X_test, Y_test = read_corpus(args.test_file, args.sentiment)
    else:
        X_test, Y_test = read_corpus(args.dev_file, args.sentiment)

    features = ""
    if args.word_count:
        X_train = add_word_count(X_train)
        X_test = add_word_count(X_test)
        features += "wc_"
    if args.lemmatize:
        X_train = lemmatize(X_train)
        X_test = lemmatize(X_test)
        features += "lemmatize_"
    if args.stem:
        X_train = stem(X_train)
        X_test = stem(X_test)
        features += "stem_"

    rangestart = args.rangestart
    rangeend = rangestart

    while rangestart <= args.rangeend:
        print(f"Running n_gram range {rangestart} -> {rangeend}")

        if args.tfidf:
            vec_name = "TF-IDF"
            # Convert the texts to vectors
            # We use a dummy function as tokenizer and preprocessor,
            # since the texts are already preprocessed and tokenized.
            # TF-IDF vectorizer
            vec = TfidfVectorizer(
                preprocessor=identity,
                tokenizer=identity,
                ngram_range=(rangestart, rangeend),
            )
        elif args.count_tf_idf:
            count = CountVectorizer(
                preprocessor=identity,
                tokenizer=identity,
                ngram_range=(rangestart, rangeend),
            )
            tf_idf = TfidfVectorizer(
                preprocessor=identity,
                tokenizer=identity,
                ngram_range=(rangestart, rangeend),
            )
            vec_name = "Count_TF-IDF"
            vec = FeatureUnion([("count", count), ("tf", tf_idf)])
        elif args.part_of_speech:
            count = CountVectorizer(
                preprocessor=identity,
                tokenizer=identity,
                ngram_range=(rangestart, rangeend),
            )
            pos = CountVectorizer(
                preprocessor=identity_string,
                tokenizer=spacy_pos,
                ngram_range=(rangestart, rangeend),
            )
            vec = FeatureUnion([("count", count), ("pos", pos)])
            vec_name = "BOW_POS"
        elif args.part_of_speech_tf_idf:
            count = CountVectorizer(
                preprocessor=identity,
                tokenizer=identity,
                ngram_range=(rangestart, rangeend),
            )
            pos = CountVectorizer(
                preprocessor=identity_string,
                tokenizer=spacy_pos,
                ngram_range=(rangestart, rangeend),
            )
            tf_idf = TfidfVectorizer(
                preprocessor=identity,
                tokenizer=identity,
                ngram_range=(rangestart, rangeend),
            )
            vec = FeatureUnion([("count", count), ("pos", pos), ("tf", tf_idf)])
            vec_name = "BOW_POS_TF-IDF"
        else:
            vec_name = "BOW"
            # Convert the texts to vectors
            # We use a dummy function as tokenizer and preprocessor,
            # since the texts are already preprocessed and tokenized.
            # Bag of words vectorizer
            vec = CountVectorizer(
                preprocessor=identity,
                tokenizer=identity,
                ngram_range=(rangestart, rangeend),
            )

        name = ""
        classifiers = []
        if args.naive_bayes:
            name = "nb"
            classifiers = [
                ("Multinomial NB", MultinomialNB()),
            ]

        if args.k_nearest_neighbour:
            name = "knn"
            classifiers = [
                ("KNN 3", KNeighborsClassifier(3)),
                ("KNN 5", KNeighborsClassifier()),
                ("KNN 8", KNeighborsClassifier(8)),
                ("KNN 3 Weighted", KNeighborsClassifier(3, weights="distance")),
                ("KNN 5 Weighted", KNeighborsClassifier(weights="distance")),
                ("KNN 8 Weighted", KNeighborsClassifier(8, weights="distance")),
            ]

        if args.support_vector_machine:
            name = "svm"
            classifiers = [
                ("LinearSVM C = 1", LinearSVC()),
                ("LinearSVM C = 0.5", LinearSVC(C=0.5)),
                ("LinearSVM C = 1.5", LinearSVC(C=1.5)),
                ("LinearSVM C = 0.75", LinearSVC(C=0.75)),
                ("LinearSVM C = 1.25", LinearSVC(C=1.25)),
                ("LinearSVM C = 2", LinearSVC(C=2)),
                ("LinearSVM C = 3", LinearSVC(C=3)),
                ("LinearSVM C = 10", LinearSVC(C=10)),
                ("LinearSVM C = 10", LinearSVC(C=100)),
                ("SVC C=1", SVC()),
                ("SVC C=0.5", SVC(C=0.5)),
                ("SVC C=1.5", SVC(C=1.5)),
            ]

        if args.decision_trees:
            name = "dt"
            classifiers = [
                (
                    "Decision Tree Entropy + Random",
                    DecisionTreeClassifier(criterion="entropy", splitter="random"),
                ),
                (
                    "Decision Tree Entorpy + Best",
                    DecisionTreeClassifier(criterion="entropy"),
                ),
                (
                    "Decision Tree Gini + Random",
                    DecisionTreeClassifier(criterion="gini", splitter="random"),
                ),
                ("Decision Tree Gini + Best", DecisionTreeClassifier(criterion="gini")),
            ]

        if args.random_forest:
            name = "rf"
            classifiers = [
                ("Random Forest Gini ", RandomForestClassifier(criterion="gini")),
                ("Random Forest Entropy", RandomForestClassifier(criterion="entropy")),
                (
                    "Random Forest Log Loss",
                    RandomForestClassifier(criterion="log_loss"),
                ),
            ]

        if args.ensemble:
            name = "ens"
            estimators = [
                ("rf", RandomForestClassifier(criterion="log_loss")),
                ("svm", LinearSVC()),
            ]
            classifiers = [
                ("Ensemble method", VotingClassifier(estimators, voting="hard"))
            ]

        if args.best:
            name = "best"
            vec_name = "TF-IDF"
            vec = TfidfVectorizer(
                preprocessor=identity,
                tokenizer=identity,
                ngram_range=(1, 1),
            )
            classifiers = [
                ("LinearSVM C = 0.75", LinearSVC(C=0.75))
            ]

        # TODO: Add comment
        df = setup_df()

        # TODO: Add comment
        results = run_experiments(
            classifiers, vec, vec_name, X_train, Y_train, X_test, Y_test
        )

        # TODO: Add comment
        df_extended = pd.DataFrame(results, columns=df.columns)
        df = pd.concat([df, df_extended])
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        df.to_excel(
            f"{args.result_dir}/{name}-{vec_name}-{features}-{rangestart}-{rangeend}.xlsx"
        )

        # TODO: Add comment
        rangeend += 1
        if rangeend > args.rangeend:
            rangestart += 1
            rangeend = rangestart
