import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

nlp = spacy.load("en_core_web_sm")


# Dummy function that only returns the input, to be used in the count functions of sklearn
def identity(inp):
    return inp


def identity_string(inp):
    """
    SpaCy can only extract the POS tags of words and not lists, therefore we need to use this function
    that joins the whole input.
    @param inp: List of words
    @return: Concatenated string of words with spaces.
    """
    return " ".join(inp)


def spacy_pos(txt):
    """
    Return the SpaCy POS tag per token
    @param txt: Concatenated string of words
    @return: list: POS tag of each word from `txt`
    """
    return [token.pos_ for token in nlp(txt)]


def get_scores(key: str, report_dict):
    """
    Get the scores of each topic from the classification report dictionary
    @param key: the key like "camera" or "dvd"
    @param report_dict: The classification report dictionary
    @return: The values of the classification report dictionary of the given key
    """
    dict_values = report_dict[key]
    return dict_values.values()


def save_confusion_matrix(Y_test, Y_pred, classes, name):
    """
    Save the confusion matrix of the specified classifier
    @param Y_test: The ground-truth Y-values
    @param Y_pred: Predicted Y-values
    @param classifier: Classifier used to get the classes
    @param name: Name of the classifier
    """
    cm = confusion_matrix(Y_test, Y_pred)
    # Plot the confusion matrix, given the confusion matrix and the classifier classes
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    disp.ax_.set_title(name)
    # Save the plot cm (confusion matrix) directory
    # Saved with name given, should be a descriptive name
    if not os.path.exists("cm"):
        os.makedirs("cm")
    disp.figure_.savefig(f"cm/{name}.png", dpi=300)
    print(f"Confusion matrix of classifier {name} is saved to cm/{name}.png")


def undersample(X_train, Y_train):
    # Split the data into arrays, one for each class
    X_train = np.array(X_train, dtype="object")
    Y_train = np.array(Y_train)
    Y_indices_off = np.where(Y_train == "OFF")[0]
    Y_indices_not = np.where(Y_train == "NOT")[0]
    X_off = X_train[Y_indices_off]
    X_not = X_train[Y_indices_not]

    # Determine the size of the smaller class
    min_size = min(len(X_off), len(X_not))

    # Random seed for reproducibility
    random_seed = 42

    # Undersample both classes to match the size of the smaller class
    X_off_undersampled = resample(X_off, replace=False, n_samples=min_size, random_state=random_seed)
    X_not_undersampled = resample(X_not, replace=False, n_samples=min_size, random_state=random_seed)

    # Combine the undersampled data from both classes
    X_final = np.concatenate([X_off_undersampled, X_not_undersampled])

    # Create corresponding labels for the undersampled data
    Y_final = ["OFF"] * min_size + ["NOT"] * min_size
    Y_final = np.array(Y_final)

    # Shuffle the data and labels
    shuffle_indices = np.random.permutation(len(X_final))
    X_final = X_final[shuffle_indices]
    Y_final = Y_final[shuffle_indices]

    return X_final, Y_final


def setup_df():
    """
    To save the results we used Pandas, we first need to set up the dataset.
    Therefore, we need to create the columns first.
    @return: Empty DataFrame with columns.
    """
    return pd.DataFrame(
        {
            "classifier": [],
            "accuracy": [],
            "macro_precision": [],
            "macro_recall": [],
            "macro_f1": [],
            "not_p": [],
            "not_r": [],
            "not_f1": [],
            "off_p": [],
            "off_r": [],
            "off_f1": [],
        }
    )


def check_balance(y):
    """
    Check the (im)balance of the dataset by plotting each label
    @param y: the topics
    """
    data = {}

    # Compute the number of occurrences
    for label in y:
        if not data.get(label):
            data[label] = 1
        data[label] += 1

    labels = data.keys()
    values = data.values()

    # Create the bar plot
    plt.bar(labels, values)

    # Add text labels at the end of each bar
    for label, value in zip(labels, values):
        plt.text(label, value, str(value), ha="center", va="bottom")

    plt.show()


def test_performance(Y_test, Y_pred, classes, name):
    # Get the performance metrics.
    acc = accuracy_score(Y_test, Y_pred)
    macro_precision = precision_score(Y_test, Y_pred, average="macro")
    macro_recall = recall_score(Y_test, Y_pred, average="macro")
    macro_f1 = f1_score(Y_test, Y_pred, average="macro")
    # Compute performance per category by using classification report of sklearn
    report_dict = classification_report(Y_test, Y_pred, output_dict=True)

    # Print classification report
    print(classification_report(Y_test, Y_pred))
    # Print any misclassified reviews, to analyze
    # cnt = 0
    # for x_test, y_test, y_pred in zip(X_test, Y_test, Y_pred):
    #     if y_test != y_pred and cnt < 10:
    #         cnt += 1
    #         print("MISSCLASSIFIED")
    #         print(" ".join(x_test), y_test, y_pred)

    # Save confusion matrix to image
    save_confusion_matrix(Y_test, Y_pred, classes, name)
    # Load score per category in coresponding variable
    b_p, b_r, b_f1, _ = get_scores("NOT", report_dict)
    c_p, c_r, c_f1, _ = get_scores("OFF", report_dict)

    # Append all results, to return, such that it can be analyzed
    return [name, acc, macro_precision, macro_recall, macro_f1, b_p, b_r, b_f1, c_p, c_r, c_f1]


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
