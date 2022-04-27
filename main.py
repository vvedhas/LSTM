import os
import re

import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from nltk.corpus import stopwords
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold

DATASET_DIR = './data/'
SAVE_DIR = './'
X = pd.read_csv(os.path.join(DATASET_DIR, 'training_set_rel3.tsv'), sep='\t', encoding='ISO-8859-1')
y = X['domain1_score']
X = X.dropna(axis=1)
X = X.drop(columns=['rater1_domain1', 'rater2_domain1'])

minimum_scores = [-1, 2, 1, 0, 0, 0, 0, 0, 0]
maximum_scores = [-1, 12, 6, 3, 3, 4, 4, 30, 60]


def create_wordlist(corpus, remove_stopwords):
    corpus = re.sub("[^a-zA-Z]", " ", corpus)
    words = corpus.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)


def create_sentences(corpus, remove_stopwords):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(corpus.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(create_wordlist(raw_sentence, remove_stopwords))
    return sentences


def featureVecs(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    num_words = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, num_words)
    return featureVec


def avgFeatureVecs(essays, model, num_features):
    counter = 0
    essayFeatureVecs = np.zeros((len(essays), num_features), dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = featureVecs(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs


def get_model():
    model = Sequential()
    model.add(LSTM(600, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 600], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model


cv = KFold(n_splits=5, shuffle=True)
results = []
y_pred_list = []
best_qwk = 0
count = 1
for traincv, testcv in cv.split(X):
    print("\n--------Fold {}--------\n".format(count))
    X_test, X_train, y_test, y_train = X.iloc[testcv], X.iloc[traincv], y.iloc[testcv], y.iloc[traincv]

    train_essays = X_train['essay']
    test_essays = X_test['essay']

    sentences = []

    for essay in train_essays:
        sentences += create_sentences(essay, remove_stopwords=True)

    num_features = 600
    min_word_count = 40
    num_workers = 16
    context = 10
    downsampling = 1e-3

    print("Training Word2Vec Model...")
    model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context,
                     sample=downsampling)

    model.init_sims(replace=True)
    model.wv.save_word2vec_format('word2vecmodel.bin', binary=True)

    clean_train_essays = []

    for corpus in train_essays:
        clean_train_essays.append(create_wordlist(corpus, remove_stopwords=True))
    trainDataVecs = avgFeatureVecs(clean_train_essays, model, num_features)

    clean_test_essays = []
    for corpus in test_essays:
        clean_test_essays.append(create_wordlist(corpus, remove_stopwords=True))
    testDataVecs = avgFeatureVecs(clean_test_essays, model, num_features)

    trainDataVecs = np.array(trainDataVecs)
    testDataVecs = np.array(testDataVecs)
    trainDataVecs = np.reshape(trainDataVecs, (trainDataVecs.shape[0], 1, trainDataVecs.shape[1]))
    testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

    lstm_model = get_model()
    lstm_model.fit(trainDataVecs, y_train, batch_size=64, epochs=10)
    # lstm_model.load_weights('./model_weights/final_lstm.h5')
    y_pred = lstm_model.predict(testDataVecs)

    y_pred = np.around(y_pred)

    result = cohen_kappa_score(y_test.values, y_pred, weights='quadratic')
    print("Kappa Score: {}".format(result))
    results.append(result)
    if result > best_qwk:
        best_qwk = result
        lstm_model.save('./model_weights/final_lstm.h5')
        print("saved best model")
    count += 1

print("Average Kappa score after a 5-fold cross validation: ", np.around(np.array(results).mean(), decimals=4))
