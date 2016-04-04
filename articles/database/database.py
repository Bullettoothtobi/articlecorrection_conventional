from __future__ import print_function

import numpy as np
from keras.preprocessing.text import Tokenizer
from math import floor

from pymongo import MongoClient
from datetime import datetime

from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors.classification import KNeighborsClassifier

from config import config

from keras.preprocessing import text, sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers.normalization import BatchNormalization

import re

from window.window import Window

from textblob.classifiers import NaiveBayesClassifier


class Database:
    """Database Class"""

    def __init__(self):
        self.settings = config.Config()
        server = self.settings.get("database", "server")
        port = self.settings.get("database", "port")
        database = self.settings.get("database", "database")
        self.client = MongoClient(server, int(port))
        self.db = self.client[database]
        self.features = []
        self.setup()

    def setup(self):
        self.client.admin.command('setParameter', textSearchEnabled=True)
        self.db["sentences"].ensure_index(
            [
                ("content", "text"),
            ],
            default_language='none',
            name="sentences_index"
        )
        self.db["words"].ensure_index(
            [
                ("word", 1),
            ],
            name="word_word_index"
        )
        self.db["words"].ensure_index(
            [
                ("index", 1),
            ],
            name="word_index_index"
        )

    def resetDb(self):
        self.db["sentences"].drop()

    def import_sentences(self, filepath):
        created = datetime.now()
        i = 0
        with open(filepath) as f:
            for line in f:
                sentence = unicode(line, "utf-8")
                inserted = self.insert_sentence(sentence, created)
                if inserted:
                    i += 1
                    print(str(i))

    def insert_sentence(self, sentence, created):
        in_db = self.db["sentences"].count(
            {
                "$text": {
                    "$search": "\"" + sentence + "\"",
                    "$language": "none"
                }
            }
        )

        if not in_db:
            self.db["sentences"].insert_one(
                {
                    "sentence": sentence,
                    "created": created
                }
            )
            return True
        else:
            return False

    def insert_word(self, word, index):
        in_db = self.db["words"].count(
            {
                "word": word
            }
        )

        if not in_db:
            self.db["words"].insert_one(
                {
                    "word": word,
                    "index": index,
                }
            )
            return True
        else:
            return False

    def get_wordcount(self):
        print(self.db["words"].count())

    def create_wordindexes(self):
        stop = False
        sentence_count = 0
        word_count = 0
        startIndex = 0
        size = 50
        stopIndex = size

        while not stop:
            sentences = self.db["sentences"].find()[startIndex:stopIndex]
            startIndex += size
            stopIndex += size
            if sentences.count() == 0:
                print("All done!")
                return True
            for sentence in sentences:
                sentence_count += 1
                if sentence_count % 1000 == 0:
                    print("sentences: ", sentence_count, "words: ", word_count)
                words = text.text_to_word_sequence(sentence["sentence"].encode("utf8"), filters=text.base_filter(),
                                                   lower=True, split=" ")
                for word in words:
                    inserted = self.insert_word(word, word_count)
                    if inserted:
                        word_count += 1

    def show_words(self):
        words = self.db["words"].find()[0:5000]
        for word in words:
            print(word["index"], ": ", word["word"])

    def get_word_from_index(self, word_id):
        return self.db["words"].find_one(
            {
                "index": word_id
            }
        )["word"]

    def get_windows(self, windowlength, windowcount):
        windows = []
        sequence = []
        articles = []
        startIndex = 0
        size = 50
        stopIndex = size
        while len(windows) < windowcount:
            sentences = self.db["sentences"].find()[startIndex:stopIndex]
            startIndex += size
            stopIndex += size
            for sentence in sentences:
                words = text.text_to_word_sequence(sentence["sentence"].encode("utf8"), filters=text.base_filter(),
                                                   lower=True, split=" ")
                for word in words:
                    # word_id = self.db["words"].find_one(
                    #         {
                    #             "word": word
                    #         }
                    # )["index"]
                    if word in ["a", "an", "the"]:
                        articles.append(word)
                    else:
                        sequence.append(word)
                        if len(sequence) == windowlength:
                            window = Window(sequence, articles, self.db)
                            windows.append(window)
                            if windowcount > 10000 and len(windows) % 10000 == 0:
                                print(len(windows), "/", windowcount)
                            if len(windows) == windowcount:
                                return windows
                            else:
                                sequence = []
                                articles = []
        return windows

    def windowing(self):

        np.random.seed(1337)  # for reproducibility
        windowlength = 6
        windowcount = 1000
        batch_size = 16
        nb_epoch = 5

        print("Initializing ", windowcount, " windows...")
        windows = self.get_windows(windowlength, windowcount)
        print(len(windows), " windows initialized.")
        # for window in windows:
        #     print(window.printable())

        X_train = []
        y_train = []
        for window in windows[0:int(round(windowcount * 0.7))]:
            text = ""
            for word in window.sequence:
                text += " " + word
            X_train.append(text)
            y_train.append(1 if len(window.articles) > 0 else 0)

        X_test = []
        y_test = []
        for window in windows[int(round(windowcount * 0.7)):windowcount + 1]:
            for word in window.sequence:
                text += " " + word
            X_test.append(text)
            y_test.append(1 if len(window.articles) > 0 else 0)

        print("Train sequences: ", len(X_train))
        print("Test sequences: ", len(X_test))

        nb_classes = np.max(y_train) + 1
        print("Classes: ", nb_classes)

        print('Vectorizing sequence data...')
        tokenizer = Tokenizer(nb_words=None)
        tokenizer.fit_on_texts((X_train + X_test))
        X_train = tokenizer.texts_to_matrix(X_train, mode='binary')
        X_test = tokenizer.texts_to_matrix(X_test, mode='binary')
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)

        print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        print('Y_train shape:', Y_train.shape)
        print('Y_test shape:', Y_test.shape)

        print('Building model...')
        model = Sequential()
        # model.add(Dense(512, input_shape=(X_train.shape[1],)))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(nb_classes))
        # model.add(Activation('softmax'))
        model.add(Embedding(windowlength, 256, input_length=X_train.shape[1]))
        model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('sigmoid'))

        # model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.compile(loss='binary_crossentropy', optimizer='rmsprop')

        model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, show_accuracy=True,
                  validation_split=0.1)
        score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

    def test(self, classifier_name="MultinomialNB"):

        if classifier_name == "MultinomialNB":
            classifier = MultinomialNB()
        elif classifier_name == "KNeighbors":
            n_neighbors = 4
            classifier = KNeighborsClassifier(n_neighbors)


        np.random.seed(1337)
        rounds = 3
        window_count = 100000
        print("Creating", window_count, "windows...")

        train_X_source, train_y_source = self.find_article_windows(3, 3, window_count)

        vectorizer = CountVectorizer(analyzer="word",
                                         tokenizer=None,
                                         preprocessor=None,
                                         stop_words=None,
                                         max_features=None)

        train_X = vectorizer.fit_transform(train_X_source).toarray()
        train_y = train_y_source

        cv = 10

        print("testing " + classifier_name + "...")
        scores = cross_validation.cross_val_score(classifier, train_X, train_y, cv=cv, scoring='accuracy', n_jobs=2)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def find_article_windows(self, word_count_previous, word_count_following, window_count):
        """Find windows surrounding an article of class a, an, the.
        :param word_count_previous: The words previous to the article within the same sentence.
        :param window_count: The number of windows to be returned.
        :param word_count_following: The words following the article within the same sentence.
        """

        start_index = 0
        size = 50
        stop_index = size
        train_X = []
        train_y = []
        articles = ["a", "an", "the"]
        while len(train_X) < window_count:
            sentences = self.db["sentences"].find()[start_index:stop_index]
            start_index += size
            stop_index += size
            for sentence in sentences:
                words = self.text2words(sentence["sentence"].encode("utf8"))
                word_count = len(words)
                for index, word in enumerate(words):
                    if word in articles \
                            and index - word_count_previous >= 0 \
                            and index + word_count_following + 1 < word_count:
                        sequence = words[index - word_count_previous: index] + \
                                   words[index + 1: index + word_count_following + 1]
                        train_X.append(" ".join(sequence))
                        train_y.append(words[index])
                        if len(train_X) == window_count:
                            return train_X, train_y
                    else:
                        sequence = words[index - word_count_previous: index + word_count_following]
                        train_X.append(" ".join(sequence))
                        train_y.append("none")
                        if len(train_X) == window_count:
                            return train_X, train_y

        return train_X, train_y

    def text2words(self, raw):
        letters_only = re.sub("[^\w]", " ", raw)
        words = letters_only.lower().split()
        return words

    def find_available_missing_article_windows(self, window_length, window_count):
        """Find windows of a specified length within sentences.
        The classes are True and False whether the first appearing artice was removed or no article was found.
        Split between classes is 50:50.
        :param window_count: The number of windows to be returned.
        :param window_length: The length of every single window.
        """

        if window_count%2 > 0:
            raise Exception("Window count needs to be even.")

        start_index = 0
        size = 50
        stop_index = size
        pos_windows = []
        neg_windows = []
        articles = ["a", "an", "the"]
        while len(pos_windows) + len(neg_windows) < window_count:
            sentences = self.db["sentences"].find()[start_index:stop_index]
            start_index += size
            stop_index += size
            for sentence in sentences:
                words = text.text_to_word_sequence(sentence["sentence"].encode("utf8"), filters=text.base_filter(),
                                                   lower=True, split=" ")
                sequence = []
                s_class = False
                for index, word_utf8 in enumerate(words):
                    word = word_utf8.decode("utf8")
                    if word in articles and not s_class:  # remove only the first article!!!
                        s_class = True
                    else:
                        sequence.append(word)

                    if len(sequence) == window_length:
                        if s_class and len(pos_windows) < window_count/2:
                            pos_windows.append([sequence, "missing_article"])
                        else:
                            if not s_class and len(neg_windows) < window_count/2:
                                neg_windows.append([sequence, "no_missing_article"])
                        sequence = []
                        s_class = False
                        if len(pos_windows) + len(neg_windows) == window_count:
                            return pos_windows + neg_windows
        return pos_windows + neg_windows

    def sentence_2_vector(self, sentence):
        vector = []
        for word in sentence:
            word_index = None
            try:
                word_index = self.features.index(word)
            except ValueError:
                self.features.append(word)
                word_index = len(word_index) - 1

            if word_index is None:
                raise Exception

            vector.append(word_index)
        return vector




        # def get_words(self, text):
        #     #return re.compile("[^\s]+").findall(text)
        #     return re.compile("\w+").findall(text)
