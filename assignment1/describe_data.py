import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lfd_assignment1 import read_corpus, create_arg_parser

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

if __name__ == "__main__":
    args = create_arg_parser()
    sns.set_theme()

    print("Data describe")

    X_train, Y_train = read_corpus(args.train_file, args.sentiment)
    check_balance(Y_train)


