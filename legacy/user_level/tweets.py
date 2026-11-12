import mysql.connector
import json

class Tweets:

    def __init__(self):
        self.mydb = mysql.connector.connect(
            host = 'localhost',
            user = 'root',
            passwd = '#password',
            database = 'canada_tweets_index3_tweets',
            collation = 'utf8mb4_unicode_ci',
            charset = 'utf8mb4'
        )

    def get_users(self, metro = None, limit = None, rand = False):
        where = ''

        if metro is not None:
            where = 'WHERE metro= "%s"' % (metro)

            if isinstance(metro, list):
                metro = "'{}'".format("','".join(metro))
                where = 'WHERE metro IN (%s)' % (metro)
        
        cursor = self.mydb.cursor()
        sql = 'SELECT * FROM USERS '

        if where:
            sql + where

        # if rand:
        # sql += ' ORDER BY RAND()'

        if limit is not None:
            sql += ' LIMIT ' + str(limit)


        cursor.execute(sql)
        result = cursor.fetchall()

        return result

    def get_tweets_by_screen_name(self, screen_name):
        cursor = self.mydb.cursor(dictionary=True)
        sql = 'SELECT * FROM Tweets WHERE screen_name= "%s"' % (screen_name)
        cursor.execute(sql)
        result = cursor.fetchall()

        return result

    def get_tweets_by_metro(self, metro, limit=None):
        where = 'WHERE metro= "%s"' % (metro)

        if isinstance(metro, list):
            metro = "'{}'".format("','".join(metro))
            where = 'WHERE metro IN (%s)' % (metro)
        
        cursor = self.mydb.cursor(dictionary=True)
        sql = 'SELECT * FROM Tweets ' + where

        if limit is not None:
            sql += 'LIMIT ' + str(limit)

        cursor.execute(sql)
        result = cursor.fetchall()

        return result


    def has_tweets(self, screen_name):
        cursor = self.mydb.cursor()
        sql = 'SELECT * FROM Tweets_preprocessed WHERE screen_name= "%s"' % (screen_name)
        cursor.execute(sql)
        result = cursor.fetchall()

        if len(result) > 0:
            return True

        return False

    def get_tweets(self):
        cursor = self.mydb.cursor(dictionary=True)
        sql = 'SELECT * FROM Tweets_preprocessed'
        cursor.execute(sql)
        result = cursor.fetchall()

        return result

    def is_tweet_added(self, tweet_id):
        cursor = self.mydb.cursor()
        sql = 'SELECT * FROM Tweets WHERE tweet_id= "%s"' % (tweet_id)
        cursor.execute(sql)
        result = cursor.fetchall()

        if len(result) > 0:
            return True

        return False


    def add_tweet(self, tweet):
        cursor = self.mydb.cursor()
        add_tweet = ("INSERT INTO Tweets "
               "(tweet_id, screen_name, text, created_at, metro, province, coordinates) "
               "VALUES (%s, %s, %s, %s, %s, %s, %s)")
        data_tweet = [tweet['tweet_id'], tweet['screen_name'], tweet['text'], tweet['created_at'], tweet['metro'], tweet['province'], json.dumps(tweet['coordinates'])]
        cursor.execute(add_tweet, data_tweet)
        self.mydb.commit()
        cursor.close()
