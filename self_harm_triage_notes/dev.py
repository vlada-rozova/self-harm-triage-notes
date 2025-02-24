from self_harm_triage_notes.config import spell_corr_dir, resources_dir
from self_harm_triage_notes.config import N_SPLITS
from self_harm_triage_notes.viz import plot_curves_cv, plot_calibration_curve
import json
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


def load_vocab(filename):
    """
    Load vocabulary.
    """
    with open (spell_corr_dir / (filename + "_vocab.json"), 'rb') as f:
        vocab = json.load(f)
        
    print("Domain-specific vocabulary contains %d words." % len(vocab))
    
    return frozenset(vocab)

def load_word_list(filename):
    """
    Load word frequency list.
    """
    with open (spell_corr_dir / (filename + "_word_freq_list.json"), 'rb') as f:
        word_list = json.load(f)
        
    print("Word frequency list contains %d unique words (%d in total)." % 
          (len(word_list), sum(word_list.values())))
    
    return Counter(word_list)

def load_misspelled_dict(filename):
    """
    Load dictionary of misspellings.
    """
    with open (spell_corr_dir / (filename + "_misspelled_dict.json"), 'rb') as f:
        misspelled_dict = json.load(f)
    
    print("Spelling correction available for %d words." % len(misspelled_dict))
        
    return misspelled_dict

def load_slang_dict():
    """
    Create a dictionary of slang used for medications mapped to their generic names.
    """
    # Load medication names
    drugs = pd.read_csv(resources_dir / "medication_names.csv")

    drugs.slang = drugs.slang.str.strip().str.lower()
    drugs.generic_name = drugs.generic_name.str.strip().str.lower()
    drugs.dropna(subset=["slang"], inplace=True)

    # Create a dictionary to convert slang to generic names
    slang_dict = dict(zip(drugs.slang, drugs.generic_name))

    print("Slang available for %d words." % len(slang_dict))

    return slang_dict

def get_stopwords():
    """
    The set of stop words when you do this:
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    ENGLISH_STOP_WORDS = set( stopwords.words('english') ).union( set(ENGLISH_STOP_WORDS) )
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
        
def get_cv_strategy(n_splits=N_SPLITS):
    """
    Return the CV object. Defaults to 5 splits.
    v1 from 13.12.23
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3)

def get_vectorizer(vectorizer_mode, params):
    """
    Call vectoriser with supplied parameters.
    """
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
    """
    Select k best features based on the chi2 statistics.
    """
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
    """
    Select features with p-value < alpha based on the chi2 statistics.
    """
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

def score_cv(model, X, y, groups=None):
    """
    Train and evaluate a model using cross-validation. 
    """
    cv = get_cv_strategy()
    
    scoring = {
        'roc' : 'roc_auc', 
        'ap' : 'average_precision'
    }
    
    start_time = time()
    
    scores = cross_validate(estimator=model, X=X, y=y, groups=groups, cv=cv, scoring=scoring, n_jobs=-1)
    
    train_time = time() - start_time
    
    print("_" * 80)
    print("Training with %d-fold cross-validation:" % cv.n_splits)
    try:
        print(model[-1])
    except:
        print(model)
    print("train time: %0.3fs" % train_time)
    print("ROC AUC score: %0.3f (+/- %0.2f)" % (scores['test_roc'].mean(), scores['test_roc'].std()))
    print("AP score: %0.3f (+/- %0.2f)" % (scores['test_ap'].mean(), scores['test_ap'].std()))
    print()

def search_params(model, search_mode, param_grid, X, y, groups=None, n_splits=N_SPLITS, refit=False, verbose=True):
    """
    Perform grid/random search to find optimal hyperparameter values.
    """
    cv = get_cv_strategy(n_splits)
    cv_generator = cv.split(X, y, groups)
    
    if search_mode=='grid':
        search = GridSearchCV(estimator=model, param_grid=param_grid, 
                              cv=cv_generator, scoring='average_precision', n_jobs=-1, 
                              refit=refit, verbose=0)
    elif search_mode=='random':
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, 
                                    cv=cv_generator, scoring='average_precision', n_jobs=-1, 
                                    refit=refit, verbose=0)
        
    search_result = search.fit(X, y)
    
    print("Best for current fold: %.3f using %s" % (search_result.best_score_, search_result.best_params_))
    
    if verbose:
        for mean, std, param in zip(search_result.cv_results_['mean_test_score'], 
                                    search_result.cv_results_['std_test_score'], 
                                    search_result.cv_results_['params']):
            print("%.3f (+/- %.3f) with: %r" % (mean, std, param))
        print()
            
    if refit:        
        return search_result.best_estimator_
    else:
        return search_result
    
def predict_cv(model, X, y, groups=None, options=[]):
    """
    Train a model and make predictions using cross-validation.
    """
    cv = get_cv_strategy()
    
    y_proba = cross_val_predict(estimator=model, X=X, y=y, groups=groups, cv=cv, method="predict_proba", n_jobs=-1)
    y_proba = y_proba[:, 1]
    
    if 'plot_curves' in options:
        cv_generator = cv.split(X, y, groups)
        plot_curves_cv(y, y_proba, cv_generator)
        
    if 'select_threshold' in options:
        cv_generator = cv.split(X, y, groups)
        select_threshold_cv(y, y_proba, cv_generator)
    
    return y_proba

def calibrate(model, X, y, y_proba):
    # Calibrated clssifier
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)

    # Model performance in each CV fold
    score_cv(calibrated_model, X, y)

    # Make predictions for each CV fold
    y_proba_calibrated = predict_cv(calibrated_model, X, y, options=['plot_curves'])

    # Plot calibration curves
    plot_calibration_curve(y, y_proba, y_proba_calibrated, filename=None)

    return calibrated_model, y_proba_calibrated

def select_threshold(y, y_proba, method='pr', beta=1.0, verbose=True):
    """
    Find optimal threshold value based ROC/PR curve.
    """ 
    if method=='roc':
        if verbose:
            print("The threshold optimises G-means calculated from the ROC curve.")
        metric = "G-means"
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        values = np.sqrt(tpr * (1-fpr))
        
              
    elif method=='pr':
        if verbose: 
            print("The threshold optimises F1-score calculated from the PR curve.")
        metric = "F1-score"
        precision, recall, thresholds = precision_recall_curve(y, y_proba)  
        values = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
        
    idx = np.argmax(values)
    thresh = thresholds[idx]
    if verbose:
        print('Best threshold for the model = %.3f, %s = %.3f' % (thresh, metric, values[idx]))
        print()
    
    eps = 0.000001
    thresh -= eps
    
    return thresh

def select_threshold_cv(y, y_proba, cv_generator, method='pr', beta=1.0, verbose=True):
    """
    Find optimal threshold for each CV fold.
    """        
    thresholds = []
    
    for _, val_idx in cv_generator:
        # Select optimal threshold
        thresh = select_threshold(y.loc[val_idx], y_proba[val_idx], method, beta, verbose)
        thresholds.append(thresh)
    
    thresholds = np.array(thresholds)
    
    print("Average optimal threshold: %0.3f (+/- %0.2f)" % (thresholds.mean(), thresholds.std()))

def threshold_proba(y_proba, thresh):
    """
    Convert predicted probabilities to crisp class labels.
    """
    assert (y_proba.min() >= 0) & (y_proba.max() <= 1)
    y_pred = np.where(y_proba > thresh, 1, 0)
    return y_pred