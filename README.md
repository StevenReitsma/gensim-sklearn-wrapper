gensim-sklearn-wrapper
======================

A scikit-learn wrapper for the gensim package. For easy usage through scikit-learn's Pipeline and GridSearchCV classes. Currently, only the transform() and fit() functions of the Latent Dirichlet Allocation (LDA) and Latent Semantic Indexing (LSI) algorithms are implemented.

Tested on:
* Python 2.7.3
* scikit-learn 0.15.2
* numpy 1.9.1
* scipy 0.14.0
* gensim 0.10.2

Pip package is not provided because it's just one file. Just download it, and import it to get started. The parameters for the class are the same as in the gensim classes themselves, so check gensim's API for parameter usage.

    from gensim_wrapper import LdaTransformer, LsiTransformer
