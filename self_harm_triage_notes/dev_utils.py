from self_harm_triage_notes.config import N_SPLITS
from collections import Counter
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import *

def get_stopwords():
    """
    The set of stop words when you do this:
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    ENGLISH_STOP_WORDS = set( stopwords.words('english') ).union( set(ENGLISH_STOP_WORDS) )
    v1 from 14.03.25
    """
    english_stopwords = [
        'a',
        'about',
        'above',
        'across',
        'after',
        'afterwards',
        'again',
        'against',
        'ain',
        'all',
        'almost',
        'alone',
        'along',
        'already',
        'also',
        'although',
        'always',
        'am',
        'among',
        'amongst',
        'amoungst',
        'amount',
        'an',
        'and',
        'another',
        'any',
        'anyhow',
        'anyone',
        'anything',
        'anyway',
        'anywhere',
        'are',
        'aren',
        'around',
        'as',
        'at',
        'back',
        'be',
        'became',
        'because',
        'become',
        'becomes',
        'becoming',
        'been',
        'before',
        'beforehand',
        'behind',
        'being',
        'below',
        'beside',
        'besides',
        'between',
        'beyond',
        'bill',
        'both',
        'bottom',
        'but',
        'by',
        'call',
        'can',
        'cannot',
        'cant',
        'co',
        'con',
        'could',
        'couldn',
        'couldnt',
        'cry',
        'd',
        'de',
        'describe',
        'detail',
        'did',
        'didn',
        'do',
        'does',
        'doesn',
        'doing',
        'don',
        'done',
        'down',
        'due',
        'during',
        'each',
        'eg',
        'eight',
        'either',
        'eleven',
        'else',
        'elsewhere',
        'empty',
        'enough',
        'etc',
        'even',
        'ever',
        'every',
        'everyone',
        'everything',
        'everywhere',
        'except',
        'few',
        'fifteen',
        'fify',
        'fill',
        'find',
        'fire',
        'first',
        'five',
        'for',
        'former',
        'formerly',
        'forty',
        'found',
        'four',
        'from',
        'front',
        'full',
        'further',
        'get',
        'give',
        'go',
        'had',
        'hadn',
        'has',
        'hasn',
        'hasnt',
        'have',
        'haven',
        'having',
        'he',
        'hence',
        'her',
        'here',
        'hereafter',
        'hereby',
        'herein',
        'hereupon',
        'hers',
        'herself',
        'him',
        'himself',
        'his',
        'how',
        'however',
        'hundred',
        'i',
        'ie',
        'if',
        'in',
        'inc',
        'indeed',
        'interest',
        'into',
        'is',
        'isn',
        'it',
        'its',
        'itself',
        'just',
        'keep',
        'last',
        'latter',
        'latterly',
        'least',
        'less',
        'll',
        'ltd',
        'm',
        'ma',
        'made',
        'many',
        'may',
        'me',
        'meanwhile',
        'might',
        'mightn',
        'mill',
        'mine',
        'more',
        'moreover',
        'most',
        'mostly',
        'move',
        'much',
        'must',
        'mustn',
        'my',
        'myself',
        'name',
        'namely',
        'needn',
        'neither',
        'never',
        'nevertheless',
        'next',
        'nine',
        'no',
        'nobody',
        'none',
        'noone',
        'nor',
        'not',
        'nothing',
        'now',
        'nowhere',
        'o',
        'of',
        'off',
        'often',
        'on',
        'once',
        'one',
        'only',
        'onto',
        'or',
        'other',
        'others',
        'otherwise',
        'our',
        'ours',
        'ourselves',
        'out',
        'over',
        'own',
        'part',
        'per',
        'perhaps',
        'please',
        'put',
        'rather',
        're',
        's',
        'same',
        'see',
        'seem',
        'seemed',
        'seeming',
        'seems',
        'serious',
        'several',
        'shan',
        'she',
        'should',
        'shouldn',
        'show',
        'side',
        'since',
        'sincere',
        'six',
        'sixty',
        'so',
        'some',
        'somehow',
        'someone',
        'something',
        'sometime',
        'sometimes',
        'somewhere',
        'still',
        'such',
        'system',
        't',
        'take',
        'ten',
        'than',
        'that',
        'the',
        'their',
        'theirs',
        'them',
        'themselves',
        'then',
        'thence',
        'there',
        'thereafter',
        'thereby',
        'therefore',
        'therein',
        'thereupon',
        'these',
        'they',
        'thick',
        'thin',
        'third',
        'this',
        'those',
        'though',
        'three',
        'through',
        'throughout',
        'thru',
        'thus',
        'to',
        'together',
        'too',
        'top',
        'toward',
        'towards',
        'twelve',
        'twenty',
        'two',
        'un',
        'under',
        'until',
        'up',
        'upon',
        'us',
        've',
        'very',
        'via',
        'was',
        'wasn',
        'we',
        'well',
        'were',
        'weren',
        'what',
        'whatever',
        'when',
        'whence',
        'whenever',
        'where',
        'whereafter',
        'whereas',
        'whereby',
        'wherein',
        'whereupon',
        'wherever',
        'whether',
        'which',
        'while',
        'whither',
        'who',
        'whoever',
        'whole',
        'whom',
        'whose',
        'why',
        'will',
        'with',
        'within',
        'without',
        'won',
        'would',
        'wouldn',
        'y',
        'yet',
        'you',
        'your',
        'yours',
        'yourself',
        'yourselves'
    ]
    return english_stopwords

def get_vectorizer(vectorizer_mode, params):
    """Call vectoriser with supplied parameters. v1 from 14.03.25"""
    if vectorizer_mode == "select features":
        return FeatureSelector(params)
    else:
        return TfidfVectorizer(analyzer=params['analyzer'], 
                               stop_words=get_stopwords(), 
                               token_pattern=r'\S+',
                               ngram_range=params['ngram_range'],
                               min_df=2, 
                               use_idf=params['use_idf'])
           
# class MeanEmbeddingVectorizer(object):
#     """
#     Class definition to instantiate a pre-trained word2vec vectoriser.
#     """
#     def __init__(self, model_path):
#         self.model_path = model_path
#         self.word2vec = gensim.models.Word2Vec.load(model_path)
#         self.dim = self.word2vec.wv.vectors[0].shape[0]

#     def fit(self, X, y=None):
#         # To add
#         return self

#     def transform(self, X):
#         tokenized_X = [doc.split() for doc in X]
                    
#         return np.array([
#             np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
#                     or [np.zeros(self.dim)], axis=0)
#             for words in tokenized_X
#         ])
    
#     def fit_transform(self, X, y=None):
#         return self.transform(X)
       
# class DocEmbeddingVectorizer(object):
    """
    Class definition to instantiate a pre-trained doc2vec vectoriser.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.doc2vec = gensim.models.Doc2Vec.load(model_path)
        self.dim = self.doc2vec.wv.vectors[0].shape[0]
    
    def fit(self, X, y=None):
        # To add
        return self
    
    def transform(self, X):
        tokenized_X = [doc.split() for doc in X]
        return np.array([
            self.doc2vec.infer_vector(words) 
            for words in tokenized_X
        ])
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
           
class FeatureSelector(object):
    """
    Class definition to instantiate a TF-IDF vectoriser with feature selection. 
    """
    def __init__(self, params):
        self.vectorizer = []
        self.analyzer = params['analyzer']
        self.ngram_range = params['ngram_range']
        self.use_idf = params['use_idf']
        self.min_df = params['min_df']
        self.mode = params['mode']
        self.thresh = params['thresh']
        self.df_features = pd.DataFrame()
        
    def fit(self, X, y):
        self.vectorizer = TfidfVectorizer(analyzer=self.analyzer, 
                                          stop_words=get_stopwords(), 
                                          token_pattern=r'\S+',
                                          ngram_range=self.ngram_range,
                                          min_df=self.min_df,
                                          use_idf=self.use_idf
                                         )
        X_ = self.vectorizer.fit_transform(X)
        feature_names = self.vectorizer.get_feature_names_out()
        
        if self.mode == "select k best":
            self.df_features = select_k_best(X_, y,
                                             feature_names, 
                                             k=self.thresh, 
                                             verbose=False)
        if self.mode == "select by pvalue":
            self.df_features = select_by_pvalue(X_, y,
                                                feature_names, 
                                                alpha=self.thresh, 
                                                verbose=False)
                                          
        self.vectorizer.set_params(vocabulary=self.df_features.feature.unique())
        self.vectorizer.fit(X, y)
                                          
        return self
        
    def transform(self, X):
        X = self.vectorizer.transform(X)
        return X
                                          
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)     
       
def select_k_best(X, y, feature_names, k=5000, verbose=False):
    """Select k best features based on the chi2 statistics. v1 from 14.03.25"""
    assert k > 1
    ch2 = SelectKBest(chi2, k=k)
    ch2.fit(X, y)
    
    df_features = pd.DataFrame({"feature" : np.asarray(feature_names)[ch2.get_support()], 
                                "score" : ch2.scores_[ch2.get_support()]})
    if verbose:
        print("Extracting %d best features by a chi-squared test..." % k)
        print("n_samples: {}, n_features: {}".format(X.shape[0], df_features.feature.nunique()))
        print()
        print("Selected features:", len(df_features))
        print("Top features:", ", ".join(df_features.sort_values(by="score").feature[:20]))
        
    return df_features

def select_by_pvalue(X, y, feature_names, alpha=0.05, verbose=False):
    """Select features with p-value < alpha based on the chi2 statistics. v1 from 14.03.25"""
    assert alpha < 1
    df_features = pd.DataFrame()
    if y.max() > 1:
        for cat in np.unique(y):
            _, p = chi2(X, y==cat)
            df_features = df_features.append(pd.DataFrame({"feature" : feature_names, 
                                                           "p_value" : p, 
                                                           "y" : cat}))
            df_features = df_features[df_features["p_value"] < alpha]

        if verbose:
            print("Extracting features by a chi-squared test with p-value < %0.2f..." % alpha)
            print("n_samples: {}, n_features: {}".format(X.shape[0], df_features.feature.nunique()))
            print()
            for cat in np.unique(y):
                print("# {}:".format(cat))
                print("Selected features:", len(df_features[df_features.y == cat]))
                print("Top features:", ", ".join(df_features.loc[df_features.y == cat].sort_values(by="p_value").feature[:20]))
                print()
    else:
        _, p = chi2(X, y)
        df_features = pd.concat([df_features, 
                                 pd.DataFrame({"feature" : feature_names, 
                                               "p_value" : p, 
                                               "y" : 1})], 
                                axis=0, ignore_index=True)
        df_features = df_features[df_features["p_value"] < alpha]
        if verbose:
            print("Extracting features by a chi-squared test with p-value < %0.2f..." % alpha)
            print("Selected features:", df_features.shape[0])
            print()
        
    return df_features