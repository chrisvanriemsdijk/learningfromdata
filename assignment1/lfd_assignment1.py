#!/usr/bin/env python

"""
This python script was used for the research paper: Exploring Classifiers for Automated Review Topic Detection

It enables the user to run experiments on the pre-defined experiments and saves the confusion matrices and results.

To use this script create a virtual environment or a Conda environment and install the necesarry packages with the
following command:
`pip install -r requirements.txt`

Also run the following command:
`python -m spacy download en_core_web_sm`

This script uses argparse, you can get all the necessary settings with:
`python lfd_assignment1.py --help`

To run our best model on the test set run the following command:
python3 lfd_assignment1.py -svm -dir results -rs 1 -re 1 -t -l -st -test test.txt

This uses the SVM experiments with the save directory set to results, range start and end of 1, lemmatizing, stemming,
and using the test file.
"""

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
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

nlp = spacy.load("en_core_web_sm")


def create_arg_parser():
    """
    Creates the argument parser with all the arguments and parses the user's input when calling the script
    @return: parsed args
    """
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
    args = parser.parse_args()
    return args


def read_corpus(corpus_file, use_sentiment):
    """
    Read the file by the given file name and parse the necesarry fields. Can use the sentiment or topic classes
    @param corpus_file: File name of the corpus_file
    @param use_sentiment: Whether to use sentiment (2-class) or topics (6-class)
    @return:
    """
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
    """
    SpaCy can only extract the POS tags of words and not lists, therefore we need to use this function
    that joins the whole input.
    @param inp: List of words
    @return: Concatenated string of words with spaces.
    """
    return " ".join(inp)


class RemoveCorrelated(BaseEstimator, TransformerMixin):
    # TODO: Comment
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
    """
    Return the SpaCy POS tag per token
    @param txt: Concatenated string of words
    @return: list: POS tag of each word from `txt`
    """
    return [token.pos_ for token in nlp(txt)]


def check_balance(y):
    """
    Chech the (im)balance of the dataset by plotting each label
    @param y: the topics
    """
    data = {}
    for label in y:
        if not data.get(label):
            data[label] = 1
        data[label] += 1
    labels = data.keys()
    values = data.values()
    plt.bar(labels, values)
    plt.show()


def get_scores(key: str, report_dict):
    """
    Get the scores of each topic from the classification report dictionary
    @param key: the key like "camera" or "dvd"
    @param report_dict: The classification report dictionary
    @return: The values of the classification report dictionary of the given key
    """
    dict_values = report_dict[key]
    return dict_values.values()


def save_confusion_matrix(Y_test, Y_pred, classifier, name):
    """
    Save the confusion matrix of the specified classifier
    @param Y_test: The ground-truth Y-values
    @param Y_pred: Predicted Y-values
    @param classifier: Classifier used to get the classes
    @param name: Name of the classifier
    """
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
    """
    Function to add word count to the input
    @param x: Features
    @return: Word count of the features
    """
    lexicon = Empath()
    n = []
    for i in x:
        add = lexicon.analyze(i, normalize=True).values()
        add = list(add)
        add = np.array(add)
        n.append(add)
    return n


def lemmatize(x):
    """
    Lemmatizes the input, goes over each word in each data sample and lemmatizes the word
    @param x: Data samples
    @return: Lemmatized data samples
    """
    lemmatizer = WordNetLemmatizer()
    new_docs = []
    for doc in x:
        new_docs.append([lemmatizer.lemmatize(word) for word in doc])
    return new_docs


def stem(x):
    """
    Stems the input, goes over each word in each data sample and stems the word
    @param x: Data samples
    @return: Stemmed data samples
    """
    stemmer = SnowballStemmer("english")
    new_docs = []
    for doc in x:
        new_docs.append([stemmer.stem(word) for word in doc])
    return new_docs


def setup_df():
    """
    To save the results we used Pandas, we first need to set up the dataset.
    Therefore, we need to create the columns first.
    @return: Empty DataFrame with columns.
    """
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
    """
    Main function to run all the classifiers with a given vectorizer, vectorizer name, and input/output.
    @param classifiers: List of classifiers to use
    @param vec: Vectorizer to use
    @param vec_name: Vectorizer name for saving in the DataFrame
    @param X_train: Train features
    @param Y_train: Train Feature labels
    @param X_test: Test features
    @param Y_test: Test feature labels
    @return: list of results
    """

    results = []
    for name, classifier in classifiers:
        pipe = []

        if not args.word_count:
            pipe.append(("vec", vec))

        if args.correlated:
            pipe.append(("corr", RemoveCorrelated()))

        pipe.append(("cls", classifier))
        pipeline = Pipeline(pipe)

        # The pipeline (vectorizer + classifier) is trained on the training features.
        pipeline.fit(X_train, Y_train)

        # Predict the labels of the development set.
        Y_pred = pipeline.predict(X_test)

        # Get the performance metrics.
        acc = accuracy_score(Y_test, Y_pred)
        macro_precision = precision_score(Y_test, Y_pred, average="macro")
        macro_recall = recall_score(Y_test, Y_pred, average="macro")
        macro_f1 = f1_score(Y_test, Y_pred, average="macro")
        report_dict = classification_report(Y_test, Y_pred, output_dict=True)

        # Print classification report
        print(classification_report(Y_test, Y_pred))

        # Print missclassified examples
        for x_test, y_test, y_pred in zip(X_test, Y_test, Y_pred):
            if y_test != y_pred:
                print("MISSCLASSIFIED")
                print(" ".join(x_test), y_test, y_pred)

        # Save confusion matrix to image
        save_confusion_matrix(Y_test, Y_pred, classifier, name)

        # Get all the scores from the classification report dictionary
        b_p, b_r, b_f1, _ = get_scores("books", report_dict)
        c_p, c_r, c_f1, _ = get_scores("camera", report_dict)
        d_p, d_r, d_f1, _ = get_scores("dvd", report_dict)
        h_p, h_r, h_f1, _ = get_scores("health", report_dict)
        m_p, m_r, m_f1, _ = get_scores("music", report_dict)
        s_p, s_r, s_f1, _ = get_scores("software", report_dict)

        # Append a row for the classifier's name and result
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

    # Read in the features and labels from the train file
    X_train, Y_train = read_corpus(args.train_file, args.sentiment)

    # Read in features from test file if we are finally ready for testing
    if args.test_file:
        X_test, Y_test = read_corpus(args.test_file, args.sentiment)
    # Otherwise load in development data
    else:
        X_test, Y_test = read_corpus(args.dev_file, args.sentiment)

    features = ""
    # Add word count
    if args.word_count:
        X_train = add_word_count(X_train)
        X_test = add_word_count(X_test)
        features += "wc_"
    # Add lemmatizing
    if args.lemmatize:
        X_train = lemmatize(X_train)
        X_test = lemmatize(X_test)
        features += "lemmatize_"
    # Add stemming
    if args.stem:
        X_train = stem(X_train)
        X_test = stem(X_test)
        features += "stem_"

    # Set range start and range end
    rangestart = args.rangestart
    rangeend = rangestart

    # To test multiple ranges we loop from rangestart till range_end
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
            # Use a feature union of BOW and TF-iDF
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
            # Use a feature union of BOW and POS
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
            # Use a feature union of BOW, POS and TF-IDf
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

        # Use the Naive Bayes classifier experiments
        if args.naive_bayes:
            name = "nb"
            classifiers = [
                ("Multinomial NB", MultinomialNB()),
            ]

        # Use the KNN classifier experiments
        if args.k_nearest_neighbour:
            name = "knn"
            classifiers = [
                ("KNN 8", KNeighborsClassifier(8)),
                ("KNN 3", KNeighborsClassifier(3)),
                ("KNN 5", KNeighborsClassifier()),
                ("KNN 3 Weighted", KNeighborsClassifier(3, weights="distance")),
                ("KNN 5 Weighted", KNeighborsClassifier(weights="distance")),
                ("KNN 8 Weighted", KNeighborsClassifier(8, weights="distance")),
            ]

        # Use the SVM classifier experiments
        if args.support_vector_machine:
            name = "svm"
            classifiers = [
                ("LinearSVM C = 0.5", LinearSVC(C=0.5)),
                ("LinearSVM C = 1", LinearSVC()),
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

        # Use the DT classifier experiments
        if args.decision_trees:
            name = "dt"
            classifiers = [
                (
                    "Decision Tree Gini + Random",
                    DecisionTreeClassifier(criterion="gini", splitter="random"),
                ),
                (
                    "Decision Tree Entropy + Random",
                    DecisionTreeClassifier(criterion="entropy", splitter="random"),
                ),
                (
                    "Decision Tree Entorpy + Best",
                    DecisionTreeClassifier(criterion="entropy"),
                ),

                ("Decision Tree Gini + Best", DecisionTreeClassifier(criterion="gini")),
            ]

        # Use the RF classifier experiments
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

        # Use the ensemble classifier experiments
        if args.ensemble:
            name = "ens"
            estimators = [
                ("rf", RandomForestClassifier(criterion="log_loss")),
                ("svm", LinearSVC()),
            ]
            classifiers = [
                ("Ensemble method", VotingClassifier(estimators, voting="hard"))
            ]

        # Set up an empty DataFrame with the needed columns
        df = setup_df()

        # Run the experiments from the user specified classifier and get the results
        results = run_experiments(
            classifiers, vec, vec_name, X_train, Y_train, X_test, Y_test
        )

        # Create a DataFrame from the results
        df_extended = pd.DataFrame(results, columns=df.columns)

        # Concatenate the empty DataFrame with the new one
        df = pd.concat([df, df_extended])

        # Create results directory if it does not exist
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

        # Save DataFrame to excel
        df.to_excel(
            f"{args.result_dir}/test-{name}-{vec_name}-{features}-{rangestart}-{rangeend}.xlsx"
        )

        # Update the range start and end accordingly to test multiple n-grams.
        rangeend += 1
        if rangeend > args.rangeend:
            rangestart += 1
            rangeend = rangestart
