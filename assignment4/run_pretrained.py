import argparse
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import random as python_random
import tensorflow as tf
from helpers.helpers_pretrained import (
    read_gpt_corpus,
    generate_tokens,
    create_pretrained,
    test_pretrained,
    train_pretrained,
    report_pretrained
)
from helpers.helpers_general import (
    read_corpus,
    lemmatize,
    stem,
    remove_emojis
)

'''Main function to train and test neural network given arguments in the dictionary'''
if __name__ == "__main__":
    np.random.seed(1234)
    tf.random.set_seed(1234)
    python_random.seed(1234)

    parser = argparse.ArgumentParser(description="Your program description here")

    parser.add_argument("--train_file", default='data/train.tsv', help="Input file to learn from")
    parser.add_argument("--dev_file", default='data/dev.tsv', help="Separate dev set to read in")
    parser.add_argument("--test_file", default='data/test.tsv', help="If added, use trained model to predict on test set")
    parser.add_argument("--gpt_file", default=False, help="If added, use GPT generated data")
    parser.add_argument("--result_dir", default='results/', help="Where to store results")
    parser.add_argument("--lemmatize", action="store_true", help="Lemmatize text")
    parser.add_argument("--stem", action="store_true", help="Stem text")
    parser.add_argument("--emoji_remove", action="store_true", help="Remove emojis from text")

    args = parser.parse_args()

    # If testing on all models, use this dict
    # Comment out models if you want to test specific models
    pretrained = {
        # "bert-base-uncased": False,
        "distilbert-base-uncased": False,
        # "roberta-large": False,
        # "cardiffnlp/twitter-xlm-roberta-base": False,
        # "GroNLP/hateBERT": True
    }

    # Read in the data and embeddings
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    X_gpt, Y_gpt = read_gpt_corpus(args.gpt_file)

    X_train = [[' '.join(subarray)] for subarray in X_train]
    X_dev = [[' '.join(subarray)] for subarray in X_dev]
    X_gpt = [[' '.join(subarray)] for subarray in X_gpt]
    # Undersampling using RandomUnderSampler

    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_train, Y_train = undersample.fit_resample(X_train, Y_train)

    if args.lemmatize:
        X_dev = lemmatize(X_dev)
        X_train = lemmatize(X_train)
        X_gpt = lemmatize(X_gpt)

    if args.stem:
        X_dev = stem(X_dev)
        X_train = stem(X_train)
        X_gpt = stem(X_gpt)

    if args.emoji_remove:
        X_dev = remove_emojis(X_dev)
        X_train = remove_emojis(X_train)
        X_gpt = remove_emojis(X_gpt)

    X_train = [text for sublist in X_train for text in sublist]
    X_dev = [text for sublist in X_dev for text in sublist]
    X_gpt = [text for sublist in X_gpt for text in sublist]

    # Transform string labels to one-hot encodings
    classes = ['OFF', 'NOT']
    Y_train_bin = np.zeros((len(Y_train), len(classes)))
    Y_dev_bin = np.zeros((len(Y_dev), len(classes)))
    Y_gpt_bin = np.zeros((len(Y_gpt), len(classes)))

    # Loop through the labels and set the corresponding class to 1
    for i, label in enumerate(Y_train):
        Y_train_bin[i, classes.index(label)] = 1

    for i, label in enumerate(Y_dev):
        Y_dev_bin[i, classes.index(label)] = 1

    for i, label in enumerate(Y_gpt):
        Y_gpt_bin[i, classes.index(label)] = 1

    # Train and test all models in the dict
    for model in pretrained:
        lm = model
        num_labels = len(set(Y_train))

        # Get tokens
        tokens_train, tokens_dev = generate_tokens(lm, X_train, X_dev)

        # Create model
        model = create_pretrained(lm, num_labels, pretrained[model])

        # Train model
        model = train_pretrained(model, tokens_train, Y_train_bin, tokens_dev, Y_dev_bin)

        # Test model
        test_pretrained(model, tokens_dev, Y_dev_bin)

        # If test_file is given, test performance on the testset
        if args.test_file:

            # Read in test set and vectorize
            X_test, Y_test = read_corpus(args.test_file)
            X_test = [[' '.join(subarray)] for subarray in X_test]
            if args.lemmatize:
                X_test = lemmatize(X_test)

            if args.stem:
                X_test = stem(X_test)

            if args.emoji_remove:
                X_test = remove_emojis(X_test)

            X_test = [text for sublist in X_test for text in sublist]

            Y_test_bin = np.zeros((len(Y_test), len(classes)))

            # Loop through the labels and set the corresponding class to 1
            for i, label in enumerate(Y_test):
                Y_test_bin[i, classes.index(label)] = 1

            tokens_test, tokens_dev = generate_tokens(lm, X_test, X_dev)

            # Finally do the predictions
            report_pretrained(model, tokens_test, Y_test_bin, X_test)

        if args.gpt_file:
            tokens_gpt, _ = generate_tokens(lm, X_gpt, X_dev)

            test_pretrained(model, tokens_gpt, Y_gpt_bin)