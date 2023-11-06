import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification

def generate_tokens(lm, X_train, X_dev, sequence_length):
  '''Given the pretrained model, generate tokens'''

  # Load pretrained tokenizer
  tokenizer = AutoTokenizer.from_pretrained(lm)

  # Tokenize sets
  tokens_train = tokenizer(X_train, padding=True, max_length=sequence_length,
    truncation=True, return_tensors="np").data
  tokens_dev = tokenizer(X_dev, padding=True, max_length=sequence_length,
    truncation=True, return_tensors="np").data
  return tokens_train, tokens_dev

def create_pretrained(lm, num_labels, pt, start_learning_rate, end_learning_rate):
  '''Given the pretrained model, return compiled model'''

  # Set polynomial decay
  decay_steps = 10000
  learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    start_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.5)

  # Set loss function and optimizer
  loss_function = BinaryCrossentropy(from_logits=True)
  optim = Adam(learning_rate=learning_rate_fn)

  # Load pretrained model and compile with the given loss and optimizer
  # pt indicates whether the pretrained model is pytorch and whether the weights should loaded from there
  model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=num_labels, from_pt=pt)
  model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
  return model

def train_pretrained(model, tokens_train, Y_train_bin, tokens_dev, Y_dev_bin, ep, batch):
  '''Given the compiled model, return a trained model'''

  # Train model
  model.fit(tokens_train, Y_train_bin, verbose=1, epochs=ep,
  batch_size=batch, validation_data=(tokens_dev, Y_dev_bin))
  return model

def read_gpt_corpus(corpus_file):
    """
    Read the file by the given file name and parse the necessary fields. Can use sentiment or topic classes.
    @param corpus_file: File name of the corpus file (CSV format).
    @return: documents (list of text) and labels (list of labels).
    """
    documents = []
    labels = []
    with open(corpus_file, encoding="utf-8") as f:
        for line in f:
            data = line.split(",")
            documents.append(",".join(data[1:-1]).strip())
            labels.append(data[-1].strip())
    return documents[1:], labels[1:]

def test_pretrained(model, tokens_dev, Y_dev_bin):
  '''Calculates accuracy given the model, tokens and true labels'''

  # Predict and get 'logits' (predicted label)
  Y_pred = model.predict(tokens_dev)["logits"]

  # Transform to label
  Y_pred = np.argmax(Y_pred, axis=1)
  Y_gold = np.argmax(Y_dev_bin, axis=1)

  # Test and print performance
  print('Accuracy on own {1} set: {0}'.format(round(accuracy_score(Y_gold, Y_pred), 3), "dev"))
  print('F1 score on own {1} set: {0}'.format(round(f1_score(Y_gold, Y_pred), 3), "dev"))

def report_pretrained(model, tokens_test, Y_test_bin, X_text):
  '''Calculates scores given the model, tokens and true labels'''

  # Predict and get 'logits' (predicted label)
  Y_pred = model.predict(tokens_test)["logits"]

  # Transform to label
  Y_pred = np.argmax(Y_pred, axis=1)
  Y_gold = np.argmax(Y_test_bin, axis=1)

  # Test and print performance
  print('Accuracy on own {1} set: {0}'.format(round(accuracy_score(Y_gold, Y_pred), 3), "test"))
  print(Y_pred)

  # Create confusion matrix
  print('Confusion matrix on own {0} set:'.format("test"))

  # Set class names and compute confusion matrix
  class_names = ["OFF", "NOT"]
  conf_matrix = confusion_matrix(Y_gold, Y_pred)

  # F1 score
  print('F1 score on own {0} set:'.format("test"))
  print(f1_score(Y_gold, Y_pred, average=None))

  # Plot confusion matrix
  plot_confusion_matrix(conf_matrix, classes=class_names)
  plt.show()

  # Print wrong (10) predicted reviews
  cnt = 0
  print('\nWrong predicted on own {0} set:'.format("test"))
  print('gold | pred | review')
  for gold, pred, text in zip(Y_gold, Y_pred, X_text):
    if gold != pred:
      cnt += 1
      print(class_names[gold], class_names[pred], text)
      if cnt == 10:
        break

  # Print correct (10) predicted reviews
  cnt = 0
  print('\nCorrect predicted on own {0} set:'.format("test"))
  print('gold | pred | review')
  for gold, pred, text in zip(Y_gold, Y_pred, X_text):
    if gold == pred:
      cnt += 1
      print(class_names[gold], class_names[pred], text)
      if cnt == 10:
        break

def plot_confusion_matrix(conf_matrix, classes):
    '''Takes a confusion matrix and class names, computes confusion matrix'''

    # Configure
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('GnBu'))
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Create plot
    fmt = 'd'
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    # Set labels
    plt.ylabel('Gold label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
