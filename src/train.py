from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
from utils import preprocess_text,tokenize_text,vec_for_learning
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.linear_model import LogisticRegression
import multiprocessing
from tqdm import tqdm
import joblib
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
cores = multiprocessing.cpu_count()



df_train=pd.read_csv('../data/train.csv')
df_train['cleaned_desc']=df_train['Exception (input)'].apply(preprocess_text)

df_test=pd.read_csv('../data/test.csv')
df_test['cleaned_desc']=df_test['Exception (input)'].apply(preprocess_text)

train_tagged = df_train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['cleaned_desc']), tags=[r['Exception Category (ouput)']]), axis=1)
test_tagged = df_test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['cleaned_desc']), tags=[r['Exception Category (ouput)']]), axis=1)

model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])
for epoch in range(30):
    model_dbow.train([x for x in tqdm(train_tagged.values)], total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

# save the model to disk
filename = '../model/dbow_model.sav'
joblib.dump(model_dbow, filename)


y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# save the model to disk
filename = '../model/classifier_model.sav'
joblib.dump(logreg, filename)

from sklearn.metrics import accuracy_score, f1_score
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
