import pandas as pd
import _pickle as cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
data = pd.read_csv('80k_new_3.csv', sep = ';')
data.drop(data.columns[[2]], axis=1, inplace=True)
y = data['moderated']
X = data['text']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train = X_train.fillna('слово')
X_test = X_test.fillna('слово')

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=100)),
     ])

text_clf.fit(X_train, y_train)

with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(text_clf, fid)