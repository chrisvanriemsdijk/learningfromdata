import argparse
import random as python_random
import numpy as np
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from helpers.helpers_baseline import (
    identity,
    identity_string,
    spacy_pos,
    setup_df,
    check_balance,
    undersample,
    test_performance,
)
from helpers.helpers_general import read_corpus, lemmatize, stem, remove_emojis


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
    # Iterate the given classifiers
    for name, classifier in classifiers:
        pipe = []

        # Append vectorizer
        pipe.append(("vec", vec))

        # Add classifier and add to sklearn Pipeline. The Pipeline keeps the code organized and overcomes mistake, such as leaking test data into training data
        pipe.append(("cls", classifier))
        pipeline = Pipeline(pipe)

        # The pipeline (vectorizer + classifier) is trained on the training features.
        pipeline.fit(X_train, Y_train)

        # Predict the labels of the development set.
        Y_pred = pipeline.predict(X_test)

        results.append(test_performance(Y_test, Y_pred, classifier.classes_, name))
    return results


def run_range(rangestart, rangeend):
    print(rangestart)
    print(rangeend)
    # Add POS to bow if indicated in arguments
    if args.part_of_speech:
        # Use a feature union of BOW and POS
        count = CountVectorizer(
            preprocessor=identity, tokenizer=identity, ngram_range=(rangestart, rangeend), min_df=min_df
        )
        pos = CountVectorizer(
            preprocessor=identity_string, tokenizer=spacy_pos, ngram_range=(rangestart, rangeend), min_df=min_df
        )
        vec = FeatureUnion([("count", count), ("pos", pos)])
        vec_name = "BOW_POS"
    else:
        vec_name = "BOW"
        # Convert the texts to vectors
        # We use a dummy function as tokenizer and preprocessor,
        # since the texts are already preprocessed and tokenized.
        # Bag of words vectorizer
        vec = CountVectorizer(
            preprocessor=identity, tokenizer=identity, ngram_range=(rangestart, rangeend), min_df=min_df
        )

    # Add the classifier indicated in the arguments
    # The classifiers will be passed to the run_experiments function, to train and test the models
    name = ""
    classifiers = []

    # Use the KNN classifier experiments
    # Init all tested KNN & SVM models
    classifiers = [
        ("KNN 3", KNeighborsClassifier(3)),
        ("KNN 5", KNeighborsClassifier()),
        ("KNN 8", KNeighborsClassifier(8)),
        ("KNN 3 Weighted", KNeighborsClassifier(3, weights="distance")),
        ("KNN 5 Weighted", KNeighborsClassifier(weights="distance")),
        ("KNN 8 Weighted", KNeighborsClassifier(8, weights="distance")),
        ("LinearSVM C = 0.5", LinearSVC(C=0.5)),
        ("LinearSVM C = 0.75", LinearSVC(C=0.75)),
        ("LinearSVM C = 1", LinearSVC()),
        ("LinearSVM C = 1.25", LinearSVC(C=1.25)),
        ("LinearSVM C = 1.5", LinearSVC(C=1.5)),
        ("LinearSVM C = 2", LinearSVC(C=2)),
        ("LinearSVM C = 3", LinearSVC(C=3)),
        ("LinearSVM C = 10", LinearSVC(C=10)),
        ("LinearSVM C = 100", LinearSVC(C=100)),
        ("SVC C=0.5", SVC(C=0.5)),
        ("SVC C=1", SVC()),
        ("SVC C=1.5", SVC(C=1.5)),
    ]

    # Setup dataframe to store the results
    df = setup_df()

    # Run the experiments for the given classifiers, vectorizer and dataset
    results = run_experiments(classifiers, vec, vec_name, X_train, Y_train, X_test, Y_test)

    # Store the results into the pandas dataframe
    df_extended = pd.DataFrame(results, columns=df.columns)

    # Concatenate the empty DataFrame with the new one
    df = pd.concat([df, df_extended])
    # Store the dataframe with results into an excel sheet, to analyze the performance easily
    # Excel name will correspond to trained model
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Save DataFrame to excel
    df.to_excel(f"{args.result_dir}/{name}-{vec_name}-{rangestart}-{rangeend}.xlsx")


if __name__ == "__main__":
    np.random.seed(1234)
    tf.random.set_seed(1234)
    python_random.seed(1234)

    parser = argparse.ArgumentParser(description="Your program description here")
    parser.add_argument("--train_file", default="data/train.tsv", help="Input file to learn from")
    parser.add_argument("--dev_file", default="data/dev.tsv", help="Separate dev set to read in")
    parser.add_argument(
        "--test_file", default="data/test.tsv", help="If added, use trained model to predict on test set"
    )
    parser.add_argument("--result_dir", default="results/", help="Where to store results")
    parser.add_argument("--part_of_speech", action="store_true", help="Define whether to use POS")
    parser.add_argument("--rangestart", type=int, default=1)
    parser.add_argument("--rangeend", type=int, default=1)
    parser.add_argument("--min_df", type=int, default=1)
    parser.add_argument("--lemmatize", action="store_true")
    parser.add_argument("--stem", action="store_true")
    parser.add_argument("--emoji_remove", action="store_true")

    args = parser.parse_args()

    # Load corpus. Use test file if given, otherwise use dev set
    X_train, Y_train = read_corpus(args.train_file)

    print("Original")
    check_balance(Y_train)

    # Undersample
    X_train, Y_train = undersample(X_train, Y_train)
    print("After undersampling")
    check_balance(Y_train)

    # Read in features from test file if we are finally ready for testing
    if args.test_file:
        X_test, Y_test = read_corpus(args.test_file)
    # Otherwise load in development data
    else:
        X_test, Y_test = read_corpus(args.dev_file)

    if args.lemmatize:
        X_test = lemmatize(X_test)
        X_train = lemmatize(X_train)

    if args.stem:
        X_test = stem(X_test)
        X_train = stem(X_train)

    if args.emoji_remove:
        X_test = remove_emojis(X_test)
        X_train = remove_emojis(X_train)

    rangestart = args.rangestart
    rangeend = args.rangestart
    min_df = args.min_df

    while rangestart <= args.rangeend:
        run_range(rangestart, rangeend)

        rangeend += 1
        if rangeend > args.rangeend:
            rangestart += 1
            rangeend = rangestart
