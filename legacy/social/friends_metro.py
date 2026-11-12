import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from tweets import Tweets
import re
import json
from geotext import GeoText
from sklearn.neural_network import MLPClassifier
from geopy.distance import geodesic
from sklearn.naive_bayes import ComplementNB


twts = Tweets()
users = twts.get_users()

with open('coordinates_metro.json') as f:
    coordinates = json.load(f)

categories = [
    'Toronto', 
    'Montreal', 
    'Vancouver', 
    'Calgary', 
    'Ottawa-Gatineau', 
    'Edmonton', 
    'Québec metro',
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

def remove_emoji(string):
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def check_metro(metro):
    if metro == 'Mississauga' or metro == 'Brampton' or metro == 'Markham' or metro == 'Vaughan':
        metro = 'Toronto'

    if metro == 'Laval' or metro == 'Longueuil':
        metro = 'Montreal'

    if metro == 'Surrey' or metro == 'Burnaby':
        metro = 'Vancouver'

    if metro == 'Ottawa' or metro == 'Gatineau':
        metro = 'Ottawa-Gatineau'

    if metro == 'Levis':
        metro = 'Quebec metro'

    if metro == 'Burlington':
        metro = 'Hamilton'

    if metro == 'Niagara Falls' or metro == 'Welland':
        metro = 'St. Catharines - Niagara'

    if metro == 'Whitby' or metro == 'Clarington':
        metro = 'Oshawa'

    if metro == 'Magog':
        metro = 'Sherbrooke'

    if metro == 'Conception Bay South' or metro == 'Mount Pearl' or metro == 'Paradise':
        metro = 'St. John\'s'

    if metro == 'Innisfil':
        metro = 'Barrie'

    if metro == 'Kelowna West':
        metro = 'Kelowna'
    
    if metro == 'South Frontenac' or metro == 'Loyalist':
        metro = 'Kingston'

    if metro == 'Dieppe' or metro == 'Riverview':
        metro = 'Moncton'

    if metro == 'Quispamsis':
        metro = 'Saint John'

    if metro == 'Selwyn':
        metro = 'Peterborough'

    if metro == 'Quinte West':
        metro = 'Belleville'

    return metro

D = {}
y = []

for category in categories:
    D[category] = 0

friends_locations = []

# users = users[:50]

for user in users:
    screen_name = user[3]
    user_location = user[9]
    friends = twts.get_friends(screen_name)

    if not friends:
        continue

    y.append(user_location)
    cpy = D

    for friend in friends:
        friend_location = remove_emoji(friend[5])        

        if friend_location:    
            places = GeoText(friend_location)
            city = places.cities

            if city:
                metro = places.cities[0]
                metro = check_metro(metro)

                if metro in categories:
                    cpy[metro] = cpy[metro] + 1

    friends_locations.append(cpy)


vec = DictVectorizer(sparse=False)
X = vec.fit_transform(friends_locations)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# parameters = {'C':[1, 10]}
# svc = SVC(kernel='rbf')
# parameters = {vim 
#     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.0001, 0.05],
#     'learning_rate': ['constant','adaptive'],
# }
mlp = MLPClassifier(max_iter=100)
# text_clf = RandomizedSearchCV(svc, parameters, cv=3)
text_clf = ComplementNB()
text_clf.fit(X_train, y_train)
# print(text_clf.best_params_)

error_distances = []
accuracy = 0
res = 0

for value, target in zip(X_test, y_test):
    predicted = text_clf.predict([value])[0]
    print(predicted)

    metro_d = coordinates[target]
    predicted_d = coordinates[predicted]
    res = geodesic((metro_d['lat'], metro_d['long']), (predicted_d['lat'], predicted_d['long'])).km
    error_distances.append(res)

    if res < 162:
        accuracy = accuracy + 1




print(np.mean(np.array(error_distances)))
print(np.median(np.array(error_distances)))
print(100 * float(accuracy)/float(len(y_test)))

# print(text_clf.score(X_test, y_test))

#     # friends_locations.append(D)


# # print(friends_locations)
