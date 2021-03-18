import argparse
from utils import preprocess_text,tokenize_text
import joblib
parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--sentence",
    help="predict Exception for the sentence",
    type=str
)
args = parser.parse_args()
if args.sentence is not None:
    raw_sent=args.sentence
    preprocessed_text=preprocess_text(raw_sent)
    tokenized_sent=tokenize_text(preprocessed_text)
    # load the model from disk
    model_dbow = joblib.load('../model/dbow_model.sav')
    X_test = model_dbow.infer_vector(tokenized_sent, steps=20)
    clf = joblib.load('../model/classifier_model.sav')
    print(clf.predict([X_test])[0])

