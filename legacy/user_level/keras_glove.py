import numpy as np
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from numpy import array
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import json
from tweets import Tweets
import random
from geopy.distance import geodesic
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
import re
import string
from nltk.corpus import stopwords


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

def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]

    return ''.join(tokens)

def load_data():
    tweets = Tweets()
    users = tweets.get_users(categories)
    random.shuffle(users)
    # users = users[:600]
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
            tweet = clean_doc(tweet)
            
    
            if tweet:
                all_tweets.append(tweet)


        all_tweets = ' '.join(all_tweets)
        X.append(all_tweets)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()
y_train = to_categorical(y_train)

t = Tokenizer()
t.fit_on_texts(X_train)
vocab_size = len(t.word_index) + 1

#ocab_size = 35
encoded_docs = [one_hot(d, vocab_size) for d in X_train]
# print(encoded_docs)
# pad documents to a max length of 4 words
# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt', mode='rt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector



max_length = 12
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define the model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=100, kernel_size=6, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(41, activation='relu'))
model.add(Dense(41, activation='softmax'))
# model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False))
# model.add(Flatten())
# model.add(Dense(41, activation='softmax'))
# model.add(Dropout(0.5))
# model.add(Dense(41, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(41, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# compile the model
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# # summarize the model
# print(model.summary())
# # # fit the model
model.fit(padded_docs, y_train, epochs=20, verbose=0, batch_size=128)


test_encoded_docs = [one_hot(d, vocab_size) for d in X_test]
test_padded_docs = pad_sequences(test_encoded_docs, maxlen=max_length, padding='post')
result = model.predict_classes(test_padded_docs)

zipped = zip(result, y_test)
error_distances = []
accuracy = 0

for predicted, target in zipped:
    metro_d = coordinates[categories[target]]
    predicted_d = coordinates[categories[predicted]]
    res = geodesic((metro_d['lat'], metro_d['long']), (predicted_d['lat'], predicted_d['long'])).km

    error_distances.append(res)

    if res < 162:
        accuracy = accuracy + 1

print(np.mean(np.array(error_distances)))
print(np.median(np.array(error_distances)))
print(100 * float(accuracy)/float(len(y_test)))