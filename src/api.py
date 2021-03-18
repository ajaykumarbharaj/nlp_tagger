import argparse
from utils import preprocess_text,tokenize_text
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# load the model from disk
model_dbow = joblib.load('../model/dbow_model.sav')
clf = joblib.load('../model/classifier_model.sav')

class Item(BaseModel):
    description: str

app = FastAPI()

@app.get("/")
def read_root():
    return {"resp": "EY GDS Hackpions 2.0"}

@app.post("/predict/")
def predict(item: Item):
    preprocessed_text=preprocess_text(item.description)
    tokenized_sent=tokenize_text(preprocessed_text)
    X_test = model_dbow.infer_vector(tokenized_sent, steps=20)
    return clf.predict([X_test])[0]



