from tweets import Tweets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from geopy.distance import geodesic
import json
import numpy as np
import preprocessor as p
import re
import enchant
from nltk.tokenize import TweetTokenizer
import random
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

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
    users = tweets.get_users(categories)
    # random.shuffle(users)
    #users = users[:1000]
    X = []
    y = []

    for user in users:
        screen_name = user[3]
        province = user[10]
        y.append(province)

        all_tweets = []
        user_tweets = tweets.get_tweets_by_screen_name(screen_name)

        for tweet in user_tweets:
            tweet = tweet['text']


            if tweet:
                # print(clean_tweet)
                all_tweets.append(tweet)


        all_tweets = ' '.join(all_tweets)
        X.append(all_tweets)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test
    # return X, y

# text_clf = Pipeline([
#     ('vect', CountVectorizer(stop_words='english')),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SGDClassifier(loss='hinge', penalty='l2',
#                            alpha=1e-3, random_state=42,
#                            max_iter=5, tol=None)),
# ])
# text_clf = Pipeline([
#     ('vect', TfidfVectorizer(stop_words='english')),
#     ('tfidf', TfidfTransformer()),
#     ('clf', LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=False,
#      intercept_scaling=1, loss='hinge', max_iter=1000,
#      multi_class='ovr', penalty='l2', random_state=1, tol=1e-05, verbose=0)),
# ])
# text_clf = Pipeline([
#     ('vect', CountVectorizer(stop_words='english')),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SVC(C=1, class_weight=None, kernel='linear', max_iter=1000, random_state=0, tol=1e-05, verbose=0)),
# ])
# text_clf = Pipeline([
#     ('vect', CountVectorizer(stop_words='english')),
#     ('tfidf', TfidfTransformer()),
#     ('clf', NuSVC(nu=0.01, kernel='linear', degree=3, gamma='auto', coef0=0.0, max_iter=-1))
# ]) 
# text_clf = Pipeline([
#     ('vect', TfidfVectorizer(stop_words='english')),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SVC(C=1.0, cache_size=1000, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
#     max_iter=-1, probability=False, random_state=None, shrinking=False,
#     tol=0.001, verbose=False)),
# ])
#alternate_sign=False
# text_clf = Pipeline([
#     ('vect', HashingVectorizer(stop_words='english', alternate_sign=False)),
#     ('tfidf', TfidfTransformer()),
#     ('clf',  MultinomialNB())
# ]) 
# text_clf = Pipeline([
#     ('vect', TfidfVectorizer(stop_words='english')),
#     ('tfidf', TfidfTransformer()),
#     ('clf',  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1))
# ]) 

text_clf = Pipeline([
    ('vect', HashingVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
    # ('to_dense', DenseTransformer()), 
    ('clf',  MLPClassifier(solver='sgd', alpha=0.0001, hidden_layer_sizes=(100,), random_state=1, max_iter=200))
]) 

class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

# text_clf = Pipeline([
#     ('vect', HashingVectorizer(stop_words='english')),
#     ('tfidf', TfidfTransformer()),
#     # ('to_dense', DenseTransformer()), 
#     ('clf',  GaussianNB())
# ]) 

X_train, X_test, y_train, y_test = load_data()
# X, y = load_data()
# kf = KFold(2, True, 1)
# kf.get_n_splits(X)

# all_t = 0

# for train_index, test_index in kf.split(X):
#     new_X = []
#     new_Y = []

#     print('train: %s, test: %s' % (len(train_index), len(test_index)))

#     for i in train_index:
#         new_X.append(X[i])
#         new_Y.append(y[i])

#     text_clf.fit(new_X, new_Y)  


#     all_ok = 0
#     known_toronto = 0
#     total_toronto = 0

#     for k in test_index:
#         X_test = X[k]
#         y_test = y[k]
#         print(y_test)

#         res = text_clf.predict([X_test])[0]

#         if y_test == 'Toronto':
#             total_toronto = total_toronto + 1

#         if res == y_test:
#             all_ok = all_ok + 1
            
#             if res == 'Toronto':
#                 known_toronto = known_toronto + 1

#     print(total_toronto, known_toronto)
#     # print('Total toronto:' + str(total_toronto) + 'and' + 'known toronto:' + str(known_toronto))

#     all_t = all_t + 100 * float(all_ok)/float(len(test_index))


# print(all_t / 10)



    # print("TRAIN:", train_index, "TEST:", test_index)


text_clf.fit(X_train, y_train)  
print('Training finished...')

# kfold = KFold(10, True, 1)

# # enumerate splits
# for train, test in kfold.split(X_train):
#     print(test)
	# print('train: %s, test: %s' % (train, test))

# print(len(y_test))

error_distances = []
accuracy = 0

for data, target in zip(X_test, y_test):
    predicted = text_clf.predict([data])[0]

    # if coordinates[target] == coordinates[predicted]:
    #     print('here')

    province_d = coordinates[target]
    predicted_d = coordinates[predicted]
    res = geodesic((province_d['lat'], province_d['long']), (predicted_d['lat'], predicted_d['long'])).km

    error_distances.append(res)

    if res < 162:
        accuracy = accuracy + 1

print(np.mean(np.array(error_distances)))
print(np.median(np.array(error_distances)))
print(100 * float(accuracy)/float(len(y_test)))
