import nltk
import re
from nltk.stem import WordNetLemmatizer

en_stop = set(nltk.corpus.stopwords.words('english'))

stemmer = WordNetLemmatizer()


def preprocess_text(str_):
    str_ = re.sub(r'\W', ' ', str(str_))  # Remove all the special characters

    str_ = re.sub(r'\s+[a-zA-Z]\s+', ' ', str_)  # remove all single characters

    str_ = re.sub(r'\^[a-zA-Z]\s+', ' ', str_)  # Remove single characters from the start
    #
    str_ = re.sub(r'\s+', ' ', str_, flags=re.I)  # Substituting multiple spaces with single space

    str_ = str_.lower()  # Converting to Lowercase

    tokens = str_.split()

    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in en_stop]
    tokens = [word for word in tokens if len(word) > 3]

    preprocessed_text_final = ' '.join(tokens)

    return preprocessed_text_final

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors