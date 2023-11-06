import random as python_random
import argparse
import numpy as np
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
from helpers.helpers_general import read_corpus, lemmatize, stem, remove_emojis
from helpers.helpers_lstm import read_embeddings, get_emb_matrix, create_model, train_model, test_set_predict

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
    parser.add_argument("--lemmatize", action="store_true", help="Lemmatize text")
    parser.add_argument("--stem", action="store_true", help="Stem text")
    parser.add_argument("--emoji_remove", action="store_true", help="Remove emojis from text")
    parser.add_argument(
        "-e",
        "--embeddings",
        default="data/glove.twitter.27B.200d.txt",
        type=str,
        help="Embedding file we are using (default glove.twitter.27B.200d.txt)",
    )
    parser.add_argument(
        "-tr",
        "--trainable",
        action="store_true",
        help="Use a trainable Embedding layer",
    )
    parser.add_argument(
        "-dense",
        "--add_dense",
        action="store_true",
        help="Apply a dense layer between the embedding layers and LSTM layers",
    )
    parser.add_argument(
        "-layers",
        "--add_layer",
        action="store_true",
        help="Create the LSTM model with two layers",
    )
    parser.add_argument(
        "-do",
        "--dropout",
        default=0.2,
        type=float,
        help="Percentage of dropout applied in the LSTM layers",
    )
    parser.add_argument(
        "-bi",
        "--bidirectional",
        action="store_true",
        help="Implement bidirectional LSTM",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.0005,
        type=float,
        help="Set learning rate",
    )
    parser.add_argument(
        "-b",
        "--batch",
        default=128,
        type=float,
        help="Set batch size",
    )

    args = parser.parse_args()

    # print("READING DATA")
    embeddings = read_embeddings(args.embeddings)
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    X_train = [[" ".join(subarray)] for subarray in X_train]
    X_dev = [[" ".join(subarray)] for subarray in X_dev]

    # Undersampling using RandomUnderSampler
    undersample = RandomUnderSampler(sampling_strategy="majority")
    X_train, Y_train = undersample.fit_resample(X_train, Y_train)

    if args.lemmatize:
        X_dev = lemmatize(X_dev)
        X_train = lemmatize(X_train)

    if args.stem:
        X_dev = stem(X_dev)
        X_train = stem(X_train)

    if args.emoji_remove:
        X_dev = remove_emojis(X_dev)
        X_train = remove_emojis(X_train)

    X_train = [text for sublist in X_train for text in sublist]
    X_dev = [text for sublist in X_dev for text in sublist]

    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)

    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)

    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    # Transform string labels to one-hot encodings
    classes = ["OFF", "NOT"]
    Y_train_bin = np.zeros((len(Y_train), len(classes)))
    Y_dev_bin = np.zeros((len(Y_dev), len(classes)))

    # Loop through the labels and set the corresponding class to 1
    for i, label in enumerate(Y_train):
        Y_train_bin[i, classes.index(label)] = 1

    for i, label in enumerate(Y_dev):
        Y_dev_bin[i, classes.index(label)] = 1

    # Create model
    model = create_model(Y_train, emb_matrix, args)

    # Transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    # Train the model
    model = train_model(model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin, args)

    if args.test_file:
        # Read in test set and vectorize
        X_test, Y_test = read_corpus(args.test_file)
        X_test = [[" ".join(subarray)] for subarray in X_test]
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
        X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
        # Finally do the predictions
        test_set_predict(model, X_test_vect, Y_test_bin, "test")
