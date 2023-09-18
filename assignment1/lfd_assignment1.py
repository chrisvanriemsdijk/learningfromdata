#!/usr/bin/env python

"""TODO: add high-level description of this Python script"""


# Import the necessary packages
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

# Download spacy data pipeline, trained on English web data
nlp = spacy.load("en_core_web_sm")

# Read arguments, such that the script can be controlled from the command line
# Define dataset to use, model to use, range to use etc.
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

# Read corpus to use as input for the sklearn models, returns array of documents and array of labels
def read_corpus(corpus_file, use_sentiment):
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

# Dummy function that only returns the input, to be used in the count functions of sklearn
def identity(inp):
    return inp

# Function that adds a space to the input and returns, to be used in the count functions of sklearn when using POS
def identity_string(inp):
    return " ".join(inp)

# Class that can be added to the sklearn Pipeline, will remove highly correlated functions
class RemoveCorrelated(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # If not list, (thus Scipy parse matrix), transfrom to dense such that it can be transformed to a pandas dataframe
        if not isinstance(X, list):
            X = X.todense()
        # Transform to pandas such that matrix operations can easily be applied
        X = pd.DataFrame(X)

        # First calculate correlation
        # Make sure each correlation appears only once in the matrix, by taking only the upperhalf of the matrix
        # A correlation is considered highly correlated if higher than 0.90, those will be dropped
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
        print(to_drop)
        # Instead of dropping the feature will become only zeros. This has the same effect as dropping the column
        for i in to_drop:
            X[i] = 0

        # Transform back to parse matrix 
        X = csr_matrix(X.values)
        return X

# Return the POS tag given by the spacy model, to be used asa feature vector
def spacy_pos(txt):
    return [token.pos_ for token in nlp(txt)]

# Calculates the number of labels per category, to analyze the data
def check_balance(y):
    data = {}
    # Compute the number of occurences
    for label in y:
        if not data.get(label):
            data[label] = 1
        data[label] += 1
    labels = data.keys()
    values = data.values()
    # Show in a plot
    plt.bar(labels, values)
    print(data)
    plt.show()

# Use a dictionary to store the scores per category
# Return the scores of a given label
def get_scores(key: str, report_dict):
    dict_values = report_dict[key]
    return dict_values.values()

# Save a confusion matrix to analyze the performance of a model
def save_confusion_matrix(Y_test, Y_pred, classifier, name):
    # Compute confusion matrix with the sklearn function
    cm = confusion_matrix(Y_test, Y_pred)
    # Plot the confusion matrix, given the confusion matrix and the classifier classes
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=classifier.classes_
    )
    disp.plot()
    disp.ax_.set_title(name)
    # Save the plot cm (confusion matrix) directory
    # Saved with name given, should be a descriptive name
    if not os.path.exists("cm"):
        os.makedirs("cm")
    disp.figure_.savefig(f"cm/{name}.png", dpi=300)
    print(f"Confusion matrix of classifier {name} is saved to cm/{name}.png")

# Uses Empath to calculate the occurence of words from categories, to be used as a feature vector
def add_word_count(x):
    # Init the lexicon
    lexicon = Empath()
    n = []
    # Iterates all words in a review
    # Calculate the normalized (such that it is comparable with other words) count per category
    for i in x:
        add = lexicon.analyze(i, normalize=True).values()
        add = list(add)
        add = np.array(add)
        n.append(add)
    return n

# Lemmatize the dataset, to preprocess the data that will be used as a feature vector
def lemmatize(x):
    # Init lemmatizer (ntlk)
    lemmatizer = WordNetLemmatizer()
    new_docs = []
    # Iterates all documents in a the given dataset
    for doc in x:
        # Iterates all words in the document and lemmatize
        new_docs.append([lemmatizer.lemmatize(word) for word in doc])
    return new_docs

# Stem the datset, to preprocess the data that will be used as a feature vector
def stem(x):
    # Init an English stemmer (nltk)
    stemmer = SnowballStemmer("english")
    new_docs = []
    # Iterates all documents in the given dataset
    for doc in x:
        # Iterates all words in the document and stem
        new_docs.append([stemmer.stem(word) for word in doc])
    return new_docs

# Setup dataframe to store and visualize the performance of a model
# The classifier name, vectorizer and accuracy should be stored
# Per label the precision, recall and f1 should be stored
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

# Takes classifiers, vectorizers and data as an input. Will train and test the models
def run_experiments(classifiers, vec, vec_name, X_train, Y_train, X_test, Y_test):
    results = []
    # Iterate the given classifiers
    for name, classifier in classifiers:
        pipe = []

        # If word count is used as a feature vector, no vectorizer is used
        if not args.word_count:
            pipe.append(("vec", vec))

        # If the argument 'correlated' is given, the higly correlated features should be removed
        if args.correlated:
            pipe.append(("corr", RemoveCorrelated()))

        # Add classifier and add to sklearn Pipeline. The Pipeline keeps the code organized and overcomes mistake, such as leaking test data into training data
        pipe.append(("cls", classifier))
        pipeline = Pipeline(pipe)

        # Fit the model to the data
        pipeline.fit(X_train, Y_train)

        # Transform data and use test data to test the performance
        Y_pred = pipeline.predict(X_test)


        # Calculate accuracy, precision, recall, f1 by using sklearn functions
        acc = accuracy_score(Y_test, Y_pred)
        macro_precision = precision_score(Y_test, Y_pred, average="macro")
        macro_recall = recall_score(Y_test, Y_pred, average="macro")
        macro_f1 = f1_score(Y_test, Y_pred, average="macro")
        # Compute performance per category by using classification report of sklearn
        report_dict = classification_report(Y_test, Y_pred, output_dict=True)
        print(classification_report(Y_test, Y_pred))
        # Print any misclassified reviews, to analyze
        for x_test, y_test, y_pred in zip(X_test, Y_test, Y_pred):
            if y_test != y_pred:
                print("MISSCLASSIFIED")
                print(" ".join(x_test), y_test, y_pred)
        save_confusion_matrix(Y_test, Y_pred, classifier, name)
        # Load score per category in coresponding variable
        b_p, b_r, b_f1, _ = get_scores("books", report_dict)
        c_p, c_r, c_f1, _ = get_scores("camera", report_dict)
        d_p, d_r, d_f1, _ = get_scores("dvd", report_dict)
        h_p, h_r, h_f1, _ = get_scores("health", report_dict)
        m_p, m_r, m_f1, _ = get_scores("music", report_dict)
        s_p, s_r, s_f1, _ = get_scores("software", report_dict)

        # Append all results, to return, such that it can be analyzed
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

    # Load corpus. Use test file if given, otherwise use dev set
    X_train, Y_train = read_corpus(args.train_file, args.sentiment)
    if args.test_file:
        X_test, Y_test = read_corpus(args.test_file, args.sentiment)
    else:
        X_test, Y_test = read_corpus(args.dev_file, args.sentiment)

    # Add additional features if indicated in the arguments
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

    # Iterate all possible ngram ranges. If rangestart is larger than the given rangeend, than all possible combinations are runned
    while rangestart <= args.rangeend:
        print(f"Running n_gram range {rangestart} -> {rangeend}")

        # If tfidf argument is given, use the TF-IDF vectorizer
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
        # Use count vectorizer and TF-IDF if indicated in arguments
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
        # Add POS to bow if indicated in arguments
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
        # Add POS to tfidf if indicated in arguments
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
        # If none given, use bag of words
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

        # Add the classifier indicated in the arguments
        # The classifiers will be passed to the run_experiments function, to train and test the models
        name = ""
        classifiers = []
        if args.naive_bayes:
            name = "nb"
            # Init all tested NB models
            classifiers = [
                ("Multinomial NB", MultinomialNB()),
            ]

        if args.k_nearest_neighbour:
            name = "knn"
            # Init all tested KNN models
            classifiers = [
                ("KNN 8", KNeighborsClassifier(8)),
                ("KNN 3", KNeighborsClassifier(3)),
                ("KNN 5", KNeighborsClassifier()),
                ("KNN 3 Weighted", KNeighborsClassifier(3, weights="distance")),
                ("KNN 5 Weighted", KNeighborsClassifier(weights="distance")),
                ("KNN 8 Weighted", KNeighborsClassifier(8, weights="distance")),
            ]

        if args.support_vector_machine:
            name = "svm"
            # Init all tested SVM models
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

        if args.decision_trees:
            name = "dt"
            # Init all tested DT models
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

        if args.random_forest:
            name = "rf"
            # Init all tested RF models
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
            # Init all tested Ensemble models
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
            # Init all the best performing model
            classifiers = [
                ("LinearSVM C = 0.75", LinearSVC(C=0.75))
            ]

        # Setup dataframe to store the results
        df = setup_df()

        # Run the experiments for the given classifiers, vectorizer and dataset
        results = run_experiments(
            classifiers, vec, vec_name, X_train, Y_train, X_test, Y_test
        )

        # Store the results into the pandas dataframe
        df_extended = pd.DataFrame(results, columns=df.columns)
        df = pd.concat([df, df_extended])
        # Store the dataframe with results into an excel sheet, to analyze the performance easily
        # Excel name will correspond to trained model
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        df.to_excel(
            f"{args.result_dir}/test-{name}-{vec_name}-{features}-{rangestart}-{rangeend}.xlsx"
        )

        # If experiments for ngram range finished, the experiments for the next possible range will be conducted
        # The range will become 1 larger, if the range becomes larger than given in the arguments, raise the start of the range
        # E.g. this will follow the structure: 1->1 1->2, 2->2. Finish
        rangeend += 1
        if rangeend > args.rangeend:
            rangestart += 1
            rangeend = rangestart
