from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import re

def read_corpus(corpus_file):
    """
    Read the file by the given file name and parse the necessary fields. Can use sentiment or topic classes.
    @param corpus_file: File name of the corpus file (TSV format).
    @return: documents (list of text) and labels (list of labels).
    """
    documents = []
    labels = []
    with open(corpus_file, encoding="utf-8") as in_file:
        for line in in_file:
            columns = line.strip().split('\t')  # Assuming columns are separated by tabs
            tokens = columns[0].strip().split()
            label = columns[1]
            documents.append(tokens)
            labels.append(label)
    return documents, labels

def lemmatize(x):
    """
    Lemmatizes the input, goes over each word in each data sample and lemmatizes the word
    @param x: Data samples
    @return: Lemmatized data samples
    """
    lemmatizer = WordNetLemmatizer()
    new_docs = []
    # Iterates all documents in a the given dataset
    for doc in x:
        # Iterates all words in the document and lemmatize
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
    # Iterates all documents in the given dataset
    for doc in x:
        # Iterates all words in the document and stem
        new_docs.append([stemmer.stem(word) for word in doc])
    return new_docs

def remove_emojis(x):
  emoji = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
                      "]+", re.UNICODE)
  new_docs = []
  for doc in x:
    new_docs.append([re.sub(emoji, '', w) for w in doc])

  return new_docs