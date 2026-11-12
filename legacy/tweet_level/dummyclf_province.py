# coding=utf-8
import numpy as np
from tweets import Tweets
from sklearn.dummy import DummyClassifier
import random
from sklearn.linear_model import SGDClassifier
import json
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split

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

def get_tweets(split=0.8):
    tweets = Tweets()
    tweets = tweets.get_tweets()
    X = []
    y = []

    for tweet in tweets:
        province = tweet['province']
        X.append(province)
        y.append(0)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = get_tweets()
total = len(x_train) + len(x_test)
print("Total number of users: " + str(total))
total_train = len(x_train)
print("Total traning users:" + str(total_train))
total_test = len(x_test)
print("Total test users:" + str(total_test))

class_labels = []
two_dimensional_values = []


classifier = DummyClassifier(strategy = 'uniform')
classifier.fit(y_train,x_train)
print('Training finished.')

test_labels = []
test_two_dimensional_values = []
accuracy = 0
test_total = len(x_test)

error_distance = 0
error_distances = []
accuracy = 0

for label in x_test:
    predicted = classifier.predict(label)[0]

    metro_d = coordinates[label]
    predicted_d = coordinates[predicted]
    res = geodesic((metro_d['lat'], metro_d['long']), (predicted_d['lat'], predicted_d['long'])).km

    if res < 162:
        accuracy = accuracy + 1

    error_distances.append(res)

mean_res = np.mean(np.array(error_distances))
median_res = np.median(np.array(error_distances))
accuracy161 = 100 * float(accuracy)/float(test_total)
print('Mean:' + str(mean_res))
print('Median:' + str(median_res))
print('ACC161:' + str(accuracy161))