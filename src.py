import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
dataset = pd.read_csv("news.csv") 
labels = dataset.label
x_tr,x_te,y_tr,y_te = train_test_split(dataset['text'], labels, train_size = 0.8, test_size = 0.2, random_state = 7)
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df = 0.7)
tfidf_tr = tfidf_vectorizer.fit_transform(x_tr)
tfidf_te = tfidf_vectorizer.transform(x_te)
p_a_c = PassiveAggressiveClassifier(max_iter = 50)
p_a_c.fit(tfidf_tr,y_tr)
y_pred = p_a_c.predict(tfidf_te)
score = accuracy_score(y_te,y_pred)
print(f"Accuracy: {round(score*100,2)}%")
print(confusion_matrix(y_te,y_pred,labels=['FAKE','REAL']))