import numpy as np
from gensim import corpora, models, matutils
from sklearn.base import BaseEstimator

class LdaTransformer(BaseEstimator):
	"""
	See http://radimrehurek.com/gensim/models/ldamodel.html for parameter usage.

	X should be a list of tokens for each document, e.g. [['This', 'is', 'document', '1'], ['Second', 'document']]
	"""
	def __init__(self, n_latent_topics = 100, use_tfidf = False, distributed = False, chunksize = 2000, passes = 1, update_every = 1, alpha = 'symmetric', eta = None, decay = 0.5, eval_every = 10, iterations = 50, gamma_threshold = 0.001):
		self.n_latent_topics = n_latent_topics
		self.distributed = distributed
		self.chunksize = chunksize
		self.passes = passes
		self.update_every = update_every
		self.alpha = alpha
		self.eta = eta
		self.decay = decay
		self.eval_every = eval_every
		self.iterations = iterations
		self.gamma_threshold = gamma_threshold
		self.use_tfidf = use_tfidf

	def transform(self, X):
		corpus = [self.dictionary.doc2bow(text) for text in X]

		if self.use_tfidf:
			corpus = self.tfidf[corpus]

		corpus_lda = self.model[corpus]
		corpus_lda_dense = matutils.corpus2dense(corpus_lda, self.n_latent_topics).T

		return corpus_lda_dense

	def fit(self, X, y=None):
		self.dictionary = corpora.Dictionary(X)
		corpus = [self.dictionary.doc2bow(text) for text in X]

		if self.use_tfidf:		
			self.tfidf = models.TfidfModel(corpus)
			corpus = self.tfidf[corpus]

		self.model = models.LdaModel(corpus, id2word = self.dictionary, num_topics = self.n_latent_topics, distributed = self.distributed, chunksize = self.chunksize, passes = self.passes, update_every = self.update_every, alpha = self.alpha, eta = self.eta, decay = self.decay, eval_every = self.eval_every, iterations = self.iterations, gamma_threshold = self.gamma_threshold)

		return self

	def get_params(self, deep = False):
		return {'n_latent_topics': self.n_latent_topics, 'distributed': self.distributed, 'chunksize': self.chunksize, 'passes': self.passes, 'update_every': self.update_every, 'alpha': self.alpha, 'eta': self.eta, 'decay': self.decay, 'eval_every': self.eval_every, 'iterations': self.iterations, 'gamma_threshold': self.gamma_threshold}

class LsiTransformer(BaseEstimator):
	"""
	See http://radimrehurek.com/gensim/models/lsimodel.html for parameter usage.

	X should be a list of tokens for each document, e.g. [['This', 'is', 'document', '1'], ['Second', 'document']]
	"""
	def __init__(self, n_latent_topics = 100, use_tfidf = True, chunksize = 20000, decay = 1.0, distributed = False, onepass = True, power_iters = 2, extra_samples = 100):
		self.n_latent_topics = n_latent_topics
		self.use_tfidf = use_tfidf
		self.chunksize = chunksize
		self.decay = decay
		self.distributed = distributed
		self.onepass = onepass
		self.power_iters = power_iters
		self.extra_samples = extra_samples

	def transform(self, X):
		corpus = [self.dictionary.doc2bow(text) for text in X]

		if self.use_tfidf:
			corpus = self.tfidf[corpus]

		corpus_lsi = self.model[corpus]
		corpus_lsi_dense = matutils.corpus2dense(corpus_lsi, self.n_latent_topics).T

		return corpus_lsi_dense

	def fit(self, X, y=None):
		self.dictionary = corpora.Dictionary(X)
		corpus = [self.dictionary.doc2bow(text) for text in X]

		if self.use_tfidf:		
			self.tfidf = models.TfidfModel(corpus)
			corpus = self.tfidf[corpus]

		self.model = models.LsiModel(corpus, id2word = self.dictionary, num_topics = self.n_latent_topics, chunksize = self.chunksize, decay = self.decay, distributed = self.distributed, onepass = self.onepass, power_iters = self.power_iters, extra_samples = self.extra_samples)

		return self

	def get_params(self, deep = False):
		return {'n_latent_topics': self.n_latent_topics, 'use_tfidf': self.use_tfidf, 'chunksize': self.chunksize, 'decay': self.decay, 'distributed': self.distributed, 'onepass': self.onepass, 'power_iters': self.power_iters, 'extra_samples': self.extra_samples}