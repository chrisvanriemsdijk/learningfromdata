from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
from sklearn.metrics import f1_score, classification_report
import tensorflow as tf
from keras.initializers import Constant


def read_embeddings(embeddings_file):
    """Read in word embeddings from file and save as numpy array"""
    dict_emb = {}
    with open(embeddings_file, "rb") as f:
        for line in f:
            fields = line.split()
            word = fields[0].decode("utf-8")
            # vector = np.fromiter((float(x) for x in fields[1:]), dtype=np.float)
            vector = np.asarray(fields[1:], dtype=float)
            dict_emb[word] = vector
    return dict_emb


def get_emb_matrix(voc, emb):
    """Get embedding matrix given vocab and the embeddings"""
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def test_set_predict(model, X_test, Y_test, ident):
    """Do predictions and measure accuracy on our own test set (that we split off train)"""
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_test, axis=1)
    print("F1 score on own {1} set: {0}".format(round(f1_score(Y_test, Y_pred), 3), ident))
    print(classification_report(Y_test, Y_pred))


def create_model(Y_train, emb_matrix, args):
    """Create the Keras model to use"""
    # Define settings, you might want to create cmd line args for them
    learning_rate = args.learning_rate if args.learning_rate else 0.0005
    loss_function = "categorical_crossentropy"
    optim = Adam(learning_rate=learning_rate)
    # optim = RMSprop(learning_rate=learning_rate)
    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = len(set(Y_train))

    # Now build the model
    model = Sequential()
    if args.trainable:
        model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix), trainable=True))
    else:
        model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix), trainable=False))
    if args.add_dense:
        model.add(Dense(embedding_dim))
    if args.layers:
        if args.bidirectional:
            model.add(
                Bidirectional(
                    LSTM(
                        embedding_dim,
                        dropout=args.dropout if args.dropout else 0.0,
                        recurrent_dropout=args.recurrent_dropout if args.recurrent_dropout else 0.0,
                        return_sequences=True,
                    )
                )
            )
        else:
            model.add(
                LSTM(
                    embedding_dim,
                    dropout=args.dropout if args.dropout else 0.0,
                    recurrent_dropout=args.recurrent_dropout if args.recurrent_dropout else 0.0,
                    return_sequences=True,
                )
            )

    if args.bidirectional:
        model.add(
            Bidirectional(
                LSTM(
                    embedding_dim,
                    dropout=args.dropout if args.dropout else 0.0,
                    recurrent_dropout=args.recurrent_dropout if args.recurrent_dropout else 0.0,
                )
            )
        )
    else:
        model.add(
            LSTM(
                embedding_dim,
                dropout=args.dropout if args.dropout else 0.0,
                recurrent_dropout=args.recurrent_dropout if args.recurrent_dropout else 0.0,
            )
        )

    # Ultimately, end with dense layer with softmax
    model.add(Dense(input_dim=embedding_dim, units=num_labels, activation="softmax"))
    # Compile model using our settings, check for accuracy
    model.compile(loss=loss_function, optimizer=optim, metrics=["accuracy"])
    print(model.summary())
    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev, args):
    """Train the model here. Note the different settings you can experiment with!"""
    verbose = 1
    batch_size = 128
    epochs = 50
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It's also possible to monitor the training loss with monitor="loss"
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    # Finally fit the model to our data
    model.fit(
        X_train,
        Y_train,
        verbose=verbose,
        epochs=epochs,
        callbacks=[callback],
        batch_size=batch_size,
        validation_data=(X_dev, Y_dev),
    )
    # Print final accuracy for the model (clearer overview)
    test_set_predict(model, X_dev, Y_dev, "dev-LSTM")
    return model
