import time, os

import pandas as pd
import numpy as np

from gensim.models import FastText


class TitleOnly:

    def __init__(self, processed_train, processed_target,
                 threshold=2, embrace=30, verbose=False, load=None):

        self.train = processed_train
        self.target = processed_target
        self.threshold = threshold
        self.embrace = embrace
        self.verbose = verbose

        self.load = load
        if self.load:
            self.model = FastText.load(self.load)


    def _make_sentence(self, df):

        ttl = df.ttl_token_freq.apply(lambda x: [e[0] for e in x])
        tag = df.ttl_token_freq.apply(lambda x: [e[0] for e in x])
        return ttl + tag


    def _to_one_string(self, lst):

        tmp = ''
        for i, word in enumerate(lst):
            if i == len(lst) - 1:
                tmp += word
            else:
                tmp += word + ' '
        return tmp


    def _get_set(self, series):

        song_set = dict()
        for lst in series:
            for songs in lst:
                try:
                    song_set[songs] += 1
                except:
                    song_set[songs] = 1

        return song_set


    def fit(self):

        if self.verbose: print("Fitting...")

        ###########################################################################
        # to Train FastText
        # 1. make corpus from preprocessed Training set titles
        # 2. train fasttext with corpus
        #
        # train_words : contains list of words, such as ['list', 'of', 'words']
        # train_str   : contains concat of words, for above case, 'list of words'
        ###########################################################################

        # contains list of truncated words
        self.train_words = self._make_sentence(self.train)
        self.target_words = self._make_sentence(self.target)

        # counting how many times unique words were used
        token_dict = self._get_set(self.train_words)
        self.token_dict = pd.Series([v for v in token_dict.values()], index=[k for k in token_dict.keys()])

        # make lists of words that are NOT going to be trained
        general = {'음악', '곡', '노래', '모음', '때'}
        not_used = set(self.token_dict[self.token_dict <= self.threshold].index)
        deletion = general.union(not_used)

        self.train_words = self.train_words.apply(lambda x: list(set(x) - deletion))
        self.train_str = self.train_words.map(self._to_one_string)

        self.target_words = self.target_words.apply(lambda x: list(set(x) - deletion))
        self.target_str = self.target_words.map(self._to_one_string)

        # train FastText
        if self.verbose: print("Training FastText...")
        if not self.load:
            self.model = FastText(self.train_words, window=2)


    def predict(self, idx):

        if self.verbose:
            print(idx, self.target.plylst_title[idx])

        ids = self.target.id[idx]
        err = False

        sims = [(i, self.model.wv.similarity(self.target_str[idx], ttl)) for i, ttl in enumerate(self.train_str)]
        result = sorted(sims, key=(lambda x: x[1]))

        # Song Part
        k = 20
        song_set = dict()
        while len(song_set) < 100:
            mask = [r[0] for r in result[::-1][:k]]
            song_set = self._get_set(self.train.loc[mask, 'songs'])
            if len(song_set) < 100:
                k += 20

        similar_songs = sorted(song_set.items(), key=(lambda x: x[1]))
        pred_songs = similar_songs[::-1][:100]
        if len(pred_songs) < 100:
            err = True
            print("ID {} has Track less than 100".format(ids))

        # Tag Part
        k = 20
        tag_set = dict()
        while len(tag_set) < self.embrace:
            mask = [r[0] for r in result[::-1][:k]]
            tag_set = self._get_set(self.train.loc[mask, 'tags'])
            if len(tag_set) < self.embrace:
                k += 20

        tag_similarity = [(t, self.model.wv.similarity(self.target_str[idx], t)) for t in tag_set.keys()]
        similar_tags = sorted(tag_similarity, key=(lambda x: x[1]))
        pred_tags = similar_tags[::-1][:10]
        if len(pred_tags) < 10:
            err = True
            print("ID {} has Tags less than 10".format(ids))

        return ids, pred_songs, pred_tags, err


    def run(self, debug=None):

        if self.verbose: print("Prediction start")

        if debug:
            _iter = self.target.index[debug[0]:debug[1]]
        else:
            _iter = self.target.index

        pred, log = [], []
        for idx in _iter:

            ids, s, t, err = self.predict(idx)
            pred.append({
                "id": ids,
                "songs": list(map(lambda x: x[0], s)),
                "tags" : list(map(lambda x: x[0], t))
            })
            if self.verbose: print(t); print('\n')

        return pred, log

    def save_model(self):
        if not os.path.isdir('./models'):
            os.mkdir('./models')

        tm = time.localtime(time.time())
        fname = time.strftime('%Y-%m-%d_%I-%M-%S-%p', tm)
        os.mkdir('./models/' + fname)
        self.model.save('./models/' + fname + '/' + fname + '.model')


if __name__ == "__main__":

    DATA_PATH = '../res/'

    preprocess = False

    if preprocess:

        from wordprocessor import WordProcessor

        wp = WordProcessor()

        # Load Original
        train = pd.read_json(DATA_PATH + 'train.json')
        target = pd.read_json(DATA_PATH + 'val.json')

        # Process
        processed_train = wp.tokenize(train)
        processed_target = wp.tokenize(target)


    else:
        # Load processed data
        processed_train = pd.read_json(DATA_PATH + 'train_token_full.json')
        target = pd.read_json(DATA_PATH + 'val_token_full.json')
        processed_target = target.loc[(np.array(list(map(len, target.songs))) == 0)
                                      & (np.array(list(map(len, target.tags))) == 0)]

    # Predict
    title_case = TitleOnly(processed_train, processed_target, verbose=True)
    title_case.fit()
    pred, log = title_case.run()
    title_case.save_model()

    from arena_util import write_json
    write_json(pred, 'title_only.json')
