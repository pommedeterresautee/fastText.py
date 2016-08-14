# fastText Model representation in Python
import numpy as np



class WordVectorModel(object):
    def __init__(self, model, words):
        self._model = model
        self.words = words
        self.dim = model.dim
        self.ws = model.ws
        self.epoch = model.epoch
        self.min_count = model.minCount
        self.neg = model.neg
        self.word_ngrams = model.wordNgrams
        self.loss_name = model.lossName.decode('utf-8')
        self.model_name = model.modelName.decode('utf-8')
        self.bucket = model.bucket
        self.minn = model.minn
        self.maxn = model.maxn
        self.lr_update_rate = model.lrUpdateRate
        self.t = model.t
        self.norm = False

    def get_vector(self, word):
        return self._model.get_vector(word, norm=self.norm)

    def set_vec_norm(self, state):
        self.norm = state

    def __getitem__(self, word):
        return self._model.get_vector(word, norm=self.norm)

    def __contains__(self, word):
        return word in self.words

    def __iter__(self):
        for word in self.words:
            yield word, self._model.get_vector(word, norm=self.norm)

    def cosine_similarity(self, first_word, second_word):
        v1 = self._model.get_vector(first_word, norm=True)
        v2 = self._model.get_vector(second_word, norm=True)
        cosine_sim = np.dot(v1, v2)
        return cosine_sim
