from tweets import Tweets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from geopy.distance import geodesic
from sklearn.base import TransformerMixin
import json
import numpy as np
import preprocessor as p
import re
import enchant
from nltk.tokenize import TweetTokenizer
import random
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix

tknzr = TweetTokenizer()

with open('coordinates_province.json') as f:
    coordinates = json.load(f)

categories = [
    'Alberta',
    'British Columbia',
    'Manitoba',
    'Newfoundland and Labrador',
    'New Brunswick',
    'Nova Scotia',
    'Ontario',
    'Ontario/Quebec',
    'Quebec',
    'Saskatchewan'
]
d = enchant.Dict("en_US")


def load_data():
    tweets = Tweets()
    tweets = tweets.get_tweets()
    # random.shuffle(tweets)
    #users = users[:1000]
    X = []
    y = []

    for tweet in tweets:
        # screen_name = user[3]
        province = tweet['province']
        print(province)
        y.append(province)
        X.append(tweet'text')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

# text_clf = Pipeline([
#     ('vect', CountVectorizer(stop_words='english')),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SGDClassifier(loss='hinge', penalty='l2',
#                            alpha=1e-3, random_state=42,
#                            max_iter=5, tol=None)),
# ])
# text_clf = Pipeline([
#     ('vect', CountVectorizer(stop_words='english')),
#     ('tfidf', TfidfTransformer()),
#     ('clf', LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
#      intercept_scaling=1, loss='hinge', max_iter=1000,
#      multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)),
# ])
# text_clf = Pipeline([
#     ('vect', HashingVectorizer(stop_words='english')),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SVC(C=1, class_weight=None, kernel='rbf', max_iter=1000, random_state=0, tol=1e-05, verbose=0)),
# ])
# text_clf = Pipeline([
#     ('vect', CountVectorizer(stop_words='english')),
#     ('tfidf', TfidfTransformer()),
#     ('clf', NuSVC(nu=0.01, kernel='rbf', degree=3, gamma='scale', coef0=0.0, max_iter=-1))
# ]) 
# text_clf = Pipeline([
#     ('vect', CountVectorizer(stop_words='english')),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SVC(C=1.0, cache_size=1000, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=False,
#     tol=0.001, verbose=False)),
# ])
# text_clf = Pipeline([
#     ('vect', CountVectorizer(stop_words='english')),
#     ('tfidf', TfidfTransformer()),
#     ('clf',  MultinomialNB())
# ]) 

class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

# text_clf = Pipeline([
#     ('vect', HashingVectorizer(stop_words='english')),
#     ('tfidf', TfidfTransformer()),
#     # ('to_dense', DenseTransformer()), 
#     ('clf',  ComplementNB())
# ]) 
# text_clf = Pipeline([
#     ('vect', CountVectorizer(stop_words='english')),
#     ('tfidf', TfidfTransformer()),
#     ('clf',  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))
# ]) 
# ('to_dense', DenseTransformer()), 
text_clf = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('clf',  MLPClassifier(solver='sgd', activation='relu', alpha=0.10, hidden_layer_sizes=(100,), random_state=1))
]) 
X_train, X_test, y_train, y_test = load_data()
# X_train = np.array(X_train)
# X_train = csr_matrix(X_train)
# X_train = csr_matrix.todense(X_train)

# X_train = X_train.todense()

text_clf.fit(X_train, y_train)  
print('Training finished...')
error_distances = []
accuracy = 0

for data, target in zip(X_test, y_test):
    predicted = text_clf.predict([data])[0]

    province_d = coordinates[target]
    predicted_d = coordinates[predicted]
    res = geodesic((province_d['lat'], province_d['long']), (predicted_d['lat'], predicted_d['long'])).km

    error_distances.append(res)

    if res < 162:
        accuracy = accuracy + 1

print(np.mean(np.array(error_distances)))
print(np.median(np.array(error_distances)))
print(100 * float(accuracy)/float(len(y_test)))
