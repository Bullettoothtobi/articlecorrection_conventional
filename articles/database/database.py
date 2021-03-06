from __future__ import print_function

import itertools
import os
from collections import Counter

import nltk
import numpy as np
from keras.preprocessing.text import Tokenizer
from math import floor

from nltk.tokenize import word_tokenize
from pymongo import MongoClient
from datetime import datetime

from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.classification import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm.classes import SVC
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.utils import random, shuffle

from config import config

from keras.preprocessing import text, sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers.normalization import BatchNormalization

from g2p_seq2seq import g2p

import matplotlib.pyplot as plt

import tensorflow as tf

import re

from window.window import Window

import cPickle as pickle

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

    def test(
            self,
            classifier_name="MultinomialNB",
            confusion=False,
            report=False,
            on_pos=False,
            app_pos=False,
            app_phoneme=False,
            no_source_word=False,
            app_position=False,
            ngrams=1,
            on_existence=False,
            two_step=False
    ):

        if two_step:
            on_existence = True

        classifiers = [
            ("MultinomialNB", MultinomialNB()),
            ("GaussianNB", GaussianNB()),
            ("KNeighbors", KNeighborsClassifier(4)),
            ("SVC", SVC(kernel="sigmoid", C=0.025)),
            ("DecisionTree", DecisionTreeClassifier(max_depth=500)),
            ("RandomForest", RandomForestClassifier(max_depth=50, n_estimators=100, max_features=100))
        ]

        classifier = None

        for name, object in classifiers:
            if name == classifier_name:
                classifier = object
                break

        ngrams = int(ngrams)
        print("ngrams:", ngrams)

        np.random.seed(1337)
        window_count = 100000
        before_article = 0
        following_article = 1
        articles = ["a", "an", "the"]
        batch_size = 16
        nb_epoch = 5

        print("Creating", window_count, "windows with " + str(before_article) + " before and " + str(
            following_article) + " words following the article.")

        train_X_source, train_y_source, train_y_2_source = self.find_article_windows(word_count_previous=before_article,
                                                                   word_count_following=following_article,
                                                                   window_count=window_count,
                                                                   articles=articles,
                                                                   on_pos=on_pos,
                                                                   app_pos=app_pos,
                                                                   app_phoneme=app_phoneme,
                                                                   no_source_word=no_source_word,
                                                                   app_position=app_position,
                                                                   ngrams=ngrams,
                                                                   on_existence=on_existence)

        print("array has", str(len(train_X_source)), "samples.")
        print("array has", str(len(train_y_source)), " labels.")

        # c = list(zip(train_X_source, train_y_source, train_y_2_source))
        #
        # np.random.shuffle(c)
        #
        # train_X_source, train_y_source, train_y_2_source = zip(*c)

        if two_step:
            train_X_source, train_y_source, train_y_2_source = shuffle(train_X_source, train_y_source, train_y_2_source, random_state=42)
        else:
            train_X_source, train_y_source = shuffle(train_X_source, train_y_source, random_state=42)

        print("Done")

        print("Examples:")
        counter = Counter()
        i=0
        for sentence, class_name in itertools.izip(train_X_source, train_y_source):
            if counter[class_name] < 3:
                print(sentence + ": " + class_name)
            counter.update([class_name])

        print()
        print(counter)
        print()

        classes = []
        for key in counter:
            classes.append(key)

        print("classes:", classes)

        vectorizer = CountVectorizer(analyzer="word",
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     max_features=None,
                                     ngram_range=(ngrams, ngrams))

        train_X = []
        if classifier_name == "keras":
            train_X = train_X_source
        else:
            train_X = vectorizer.fit_transform(train_X_source).toarray()
        train_y = train_y_source
        train_y_2 = train_y_2_source

        cv = 10

        class_names = ["article", "none"] if on_existence else articles + ["none"]

        if two_step:
            test_X_source, train_X_new, train_y_new, train_y_2_new, test_x_new, test_y_new, test_y_2_new = self.split(train_X_source, train_X, train_y, train_y_2, 0.33)
        else:
            train_X_new, test_x_new, train_y_new, test_y_new = train_test_split(train_X, train_y, test_size=0.33)

        if classifier_name == "all":
            for name, classifier in classifiers:

                if confusion:
                    print("Creating confusion matrix for " + name + "...")
                    self.print_confusion_matrix(classifier=classifier, class_names=class_names,
                                                classifier_name=name, train_X=train_X_new,
                                                train_y=train_y_2_new)
                else:
                    print("Testing " + name + "...")
                    # scores = cross_validation.cross_val_score(classifier, train_X, train_y,
                    #                                           cv=cv, scoring='f1_weighted', n_jobs=1,)
                    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
                    pred_y = classifier.fit(train_X_new, train_y_new).predict(test_x_new)
                    print(classification_report(y_true=test_y_new, y_pred=pred_y, target_names=class_names))
        elif classifier_name == "keras":

            print("Testing " + classifier_name + "...")

        else:
            if confusion:
                self.print_confusion_matrix(classifier=classifier, class_names=class_names,
                                            classifier_name=classifier_name, train_X=train_X_new,
                                            train_y=train_y_new)
            else:
                print("Testing " + classifier_name + "...")
                # scores = cross_validation.cross_val_score(classifier, train_X, train_y,
                #                                           cv=cv, scoring='f1_weighted', n_jobs=1)
                # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

                print("train_X_new:", len(train_X_new))
                print("train_y_new:", len(train_y_new))

                pred_y = classifier.fit(train_X_new, train_y_new).predict(test_x_new)

                # falsecount = 0
                # for x, p, y in zip(test_X_source, pred_y, test_y_new):
                #     if p != y:
                #         falsecount += 1
                #         print(x, p, y)
                #         if falsecount == 20:
                #             break

                print("1. Prediction:") if two_step else False
                print(classification_report(y_true=test_y_new, y_pred=pred_y, target_names=class_names))



                if two_step:
                    print("2. Classify previous positive predictions")

                    class_names = articles + ["none"]

                    train_X_2 = []
                    test_x_2 = []
                    train_y_2 = []
                    test_y_2 = []
                    for pred_y_item, train_X_item, test_x_item, train_y_2_item, test_y_2_item in \
                            zip(pred_y, train_X_new, test_x_new, train_y_2_new, test_y_2_new):
                        if pred_y_item == "article":
                            train_X_2.append(train_X_item)
                            test_x_2.append(test_x_item)
                            train_y_2.append(train_y_2_item)
                            test_y_2.append(test_y_2_item)
                    print("Count: " + str(len(train_y_2)))
                    pred_y_2 = classifier.fit(train_X_2, train_y_2).predict(test_x_2)
                    print(classification_report(y_true=test_y_2, y_pred=pred_y_2))

    def split(self, test_X_source, train_X, train_y, train_y_2, test_split=0.33):
        train_X_new = []
        test_x_new = []
        train_y_new = []
        test_y_new = []
        train_y_2_new = []
        test_y_2_new = []
        test_X_source_new = []

        train_len = round(len(train_X)*(1-test_split), 0)

        for test_X_source_item, train_X_item, train_y_item, train_y_2_item in zip(test_X_source, train_X, train_y, train_y_2):
            if len(train_X_new) < train_len:
                train_X_new.append(train_X_item)
                train_y_new.append(train_y_item)
                train_y_2_new.append(train_y_2_item)
                test_X_source_new.append(test_X_source_item)
            else:
                test_x_new.append(train_X_item)
                test_y_new.append(train_y_item)
                test_y_2_new.append(train_y_2_item)

        return test_X_source_new, train_X_new, train_y_new, train_y_2_new, test_x_new, test_y_new, test_y_2_new

    def print_confusion_matrix(self, classifier, classifier_name, class_names, train_X, train_y):
        train_X, test_x, train_y, test_y = train_test_split(train_X, train_y, random_state=0)
        pred_y = classifier.fit(train_X, train_y).predict(test_x)
        cm = confusion_matrix(test_y, pred_y)
        np.set_printoptions(precision=2)
        print('Confusion matrix, without normalization')
        print(cm)
        plt.figure()
        title = "Confusion Matrix"
        self.plot_confusion_matrix(cm, class_names, title=title)
        print("save")
        path = "images/"
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
        plt.savefig(path + "confusion_" + classifier_name + ".png")
        print("done")

    def plot_confusion_matrix(self, cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.ylabel('Klasse')
        plt.xlabel('Vorhersage')
        plt.tight_layout(h_pad=20)

    def find_article_windows(
            self,
            word_count_previous,
            word_count_following,
            window_count,
            articles,
            on_pos=False,
            app_pos=False,
            app_phoneme=False,
            no_source_word=False,
            app_position=False,
            ngrams=1,
            on_existence=False
    ):
        """Find windows surrounding an article of class a, an, the and none.
        :param word_count_previous: The words previous to the article within the same sentence.
        :param window_count: The number of windows to be returned.
        :param word_count_following: The words following the article within the same sentence.
        """

        start_index = 0
        size = 50
        stop_index = size
        train_X = []
        train_y = []
        train_y_2 = []

        class_size = round(window_count / 4)

        if app_phoneme:
            print("starting theano session")
            sess = tf.Session()
            g2p.FLAGS.model="data/g2p-seq2seq-cmudict"
            gr_vocab, rev_ph_vocab, model = g2p.get_vocabs_load_model(sess=sess)
            arpabet = nltk.corpus.cmudict.dict()

        print("building class support of", class_size, "each to get", window_count, "samples.")

        counter = Counter()

        new_words = {}
        if os.path.isfile("data/new_words.p"):
            new_words = pickle.load(open("data/new_words.p", "rb"))
        new_words_count = len(new_words)

        while len(train_X) < window_count:
            sentences = self.db["sentences"].find()[start_index:stop_index]
            start_index += size
            stop_index += size
            for sentence in sentences:

                words = self.text2words(sentence["sentence"].encode("utf8"))
                pos_words = [x[1] for x in nltk.pos_tag(words, "universal")] if on_pos is True or app_pos is True else None

                phoneme = []
                if app_phoneme:
                    for word in words:
                        if word.isdigit():
                            phoneme.append("-")
                        elif word in arpabet:
                            phoneme.append(arpabet[word][0][0])
                        elif word in new_words:
                            phoneme.append(new_words[word])
                        else:
                            new_words[word] = g2p.decode_word(word, sess=sess, model=model, gr_vocab=gr_vocab, rev_ph_vocab=rev_ph_vocab).split(" ")[0];
                            phoneme.append(new_words[word])
                            new_words_count += 1
                            print("new (" + str(new_words_count) + "):", word, "-", new_words[word])
                            if new_words_count % 100:
                                pickle.dump(new_words, open("data/new_words.p", "wb"))

                word_count = len(words)
                position = 0
                for index, word in enumerate(words):
                    position += 1
                    if word in articles \
                            and index - word_count_previous*ngrams >= 0 \
                            and index + word_count_following*ngrams + 1 < word_count:
                        sequence = []
                        if on_pos:
                            sequence = pos_words[index - word_count_previous*ngrams: index] + \
                                       pos_words[index + 1: index + word_count_following*ngrams + 1]
                        elif not no_source_word:
                            sequence = words[index - word_count_previous*ngrams: index] + \
                                       words[index + 1: index + word_count_following*ngrams + 1]
                        if app_pos:
                            sequence = sequence + pos_words[index - word_count_previous*ngrams: index] + \
                                       pos_words[index + 1: index + word_count_following*ngrams + 1]

                        if app_phoneme:
                            sequence = sequence + phoneme[index - word_count_previous*ngrams: index] + \
                                       phoneme[index + 1: index + word_count_following*ngrams + 1]

                        if app_position:
                            sequence = sequence + [str(position)]

                        if counter[words[index]] < class_size:
                            counter.update([words[index]])
                            train_X.append(" ".join(sequence))
                            if on_existence:
                                train_y.append("article")
                                train_y_2.append(words[index])
                            else:
                                train_y.append(words[index])
                            if len(train_X) == window_count:
                                if new_words_count > 0:
                                    pickle.dump(new_words, open("data/new_words.p", "wb"))
                                return train_X, train_y, train_y_2
                    else:
                        sequence = []
                        if on_pos:
                            sequence = pos_words[index - word_count_previous*ngrams: index + word_count_following*ngrams]
                        elif not no_source_word:
                            sequence = words[index - word_count_previous*ngrams: index + word_count_following*ngrams]
                        if app_pos:
                            sequence = sequence + pos_words[index - word_count_previous*ngrams: index + word_count_following*ngrams]
                        if app_phoneme:
                            sequence = sequence + phoneme[index - word_count_previous*ngrams: index + word_count_following*ngrams]
                        if app_position:
                            sequence = sequence + ["-"]
                        if counter["none"] < class_size:
                            counter.update(["none"])
                            train_X.append(" ".join(sequence))
                            train_y.append("none")
                            train_y_2.append("none")
                            if len(train_X) == window_count:
                                if new_words.len > 0:
                                    pickle.dump(new_words, open("data/new_words.p", "wb"))
                                return train_X, train_y, train_y_2

        if new_words_count > 0:
            pickle.dump(new_words, open("data/new_words.p", "wb"))
        return train_X, train_y, train_y_2

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

        if window_count % 2 > 0:
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
                        if s_class and len(pos_windows) < window_count / 2:
                            pos_windows.append([sequence, "missing_article"])
                        else:
                            if not s_class and len(neg_windows) < window_count / 2:
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
