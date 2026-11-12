import numpy as np
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from numpy import array
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import json
from tweets import Tweets
import random
from geopy.distance import geodesic
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

with open('coordinates_metro.json') as f:
    coordinates = json.load(f)

categories = [
    'Toronto', 
    'Montreal', 
    'Vancouver', 
    'Calgary', 
    'Ottawa-Gatineau', 
    'Edmonton', 
    'Quebec City',
    'Winnipeg', 
    'Hamilton',
    'Kitchener-Cambridge-Waterloo',
    'London',
    'St. Catharines - Niagara',
    'Halifax',
    'Oshawa',
    'Victoria',
    'Windsor',
    'Saskatoon',
    'Regina',
    'Sherbrooke',
    "St. John's",
    'Barrie',
    'Kelowna',
    'Abbotsford–Mission',
    'Greater Sudbury',
    'Kingston',
    'Saguenay',
    'Trois-Rivieres',
    'Guelph',
    'Moncton',
    'Brantford',
    'Saint John',
    'Peterborough',
    'Thunder Bay',
    'Lethbridge',
    'Nanaimo',
    'Kamloops',
    'Belleville',
    'Chatham-Kent',
    'Fredericton',
    'Chilliwack',
    'Red Deer'
]

tweets = Tweets()
users = tweets.get_users(categories)
# random.shuffle(users)
users = users[:100]
X = []
y = []

for user in users:
    screen_name = user[3]
    metro = user[9]
    y.append(categories.index(metro))

    all_tweets = []
    user_tweets = tweets.get_tweets_by_screen_name(screen_name)

    for tweet in user_tweets:
        tweet = tweet['text']

        if tweet:
            all_tweets.append(tweet)


    all_tweets = ' '.join(all_tweets)
    X.append(all_tweets)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = to_categorical(y_train)


def create_model(optimizer = 'adam', epochs=10, batch_size=10):

    # integer encode the documents
    vocab_size = 32
    encoded_docs = [one_hot(d, vocab_size) for d in X_train]
    # print(encoded_docs)
    # pad documents to a max length of 4 words
    max_length = 18
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(Flatten())
    model.add(Dense(41, activation='relu'))
    model.add(Dense(41, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

model = KerasClassifier(build_fn=create_model, verbose=0)
param_grid = {
    'batch_size': [10, 20, 40, 60, 80, 100],
    'epochs': [10, 50, 100],
    'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
text_clf = grid.fit(X_train, y_train)

print('Best parameters found:\n', text_clf.best_params_)


# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# # compile the model
# model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# # # summarize the model
# # print(model.summary())
# # # # fit the model
# model.fit(padded_docs, y_train, epochs=20, verbose=0, batch_size=64)


# test_encoded_docs = [one_hot(d, vocab_size) for d in X_test]

# test_padded_docs = pad_sequences(test_encoded_docs, maxlen=max_length, padding='post')
# result = model.predict_classes(test_padded_docs)

# zipped = zip(result, y_test)
# error_distances = []
# accuracy = 0

# for predicted, target in zipped:
#     metro_d = coordinates[categories[target]]
#     predicted_d = coordinates[categories[predicted]]
#     res = geodesic((metro_d['lat'], metro_d['long']), (predicted_d['lat'], predicted_d['long'])).km

#     error_distances.append(res)

#     if res < 162:
#         accuracy = accuracy + 1

# print(np.mean(np.array(error_distances)))
# print(np.median(np.array(error_distances)))
# print(100 * float(accuracy)/float(len(y_test)))