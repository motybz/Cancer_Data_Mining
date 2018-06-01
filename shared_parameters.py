from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing
from sklearn.metrics import *

from copy import deepcopy
import pickle,os
import pandas as pd

# Functions
def test_the_data(X_test, Y_test, fited_model,scoring):
    # Make predictions on test dataset
    scores = []
    scorer = get_scorer(scoring)
    score = scorer(fited_model,X_test, Y_test)
    scores.append(score)
    return scores

# Classes
class DataSet:
    def __init__(self, X, Y, feature_list,X_encoding_dict,Y_encoding_dict, threshold=None, ratio=None):
        self.X = X
        self.X_encoders = X_encoding_dict
        self.Y = Y
        self.Y_encoder = Y_encoding_dict
        self.feature_list = feature_list
        self.threshold = threshold
        self.ratio = ratio

class SkFModel:
    def __init__(self, name, sk_model):
        self.name = name
        self.skmodel = sk_model

class TrainedModel:
    def __init__(self, train_set=DataSet, my_sk_model=SkFModel, train_score=float ,test_set=DataSet):
        self.train_set = train_set
        self.test_set = test_set
        self.chosen_model = my_sk_model
        self.fited_model = None
        self.train_score = train_score
        self.test_score = None

    def test_the_model(self,scoring):
        self.test_score = test_the_data(self.test_set.X, self.test_set.Y, self.fited_model,scoring)
        return self.test_score

    def train_model(self):
        clf = self.chosen_model.skmodel
        clf.fit(self.train_set.X, self.train_set.Y)
        self.fited_model = deepcopy(clf)

    def save_traind_model_to_file(self,outdir):
        print('Saving model {} with {} features to file...'.format(self.chosen_model.name,len(self.train_set.feature_list)))
        filename = '{}_{}_features.sav'.format(self.chosen_model.name,len(self.train_set.feature_list))
        outdir = os.path.join(outdir,filename)
        pickle.dump(self, open(outdir, 'wb'))
        print('File ready here {}'.format(outdir))

def pre_processing(X_set, Y_set ,X_encoders = None,Y_encoder =None):
    # Data pre processing
    # Encoding Categorial features and imputing NaN's
    # https://chrisalbon.com/machine_learning/preprocessing_structured_data/convert_pandas_categorical_column_into_integers_for_scikit-learn/
    # http://pbpython.com/categorical-encoding.html
    # https://datascience.stackexchange.com/questions/14069/mass-convert-categorical-columns-in-pandas-not-one-hot-encoding

    # String categories to int
    if X_encoders:
        for feature, le in X_encoders.items():
            if feature in X_set.columns:
                X_set[feature][pd.isnull(X_set[feature])] = 'NaN'
                X_set[feature] = le.transform(X_set[feature])
    else:
        char_cols = X_set.dtypes.pipe(lambda x: x[x == 'object']).index
        X_encoders = {}
        for feature in char_cols:
            # https://stackoverflow.com/questions/36808434/label-encoder-encoding-missing-values
            X_set[feature][pd.isnull(X_set[feature])] = 'NaN'
            le = preprocessing.LabelEncoder()
            le.fit(X_set[feature])
            X_set[feature] = le.transform(X_set[feature])
            X_encoders.update({feature:le})
    # Also for Y set
    if Y_encoder:
        Y_set = Y_encoder.transform(Y_set)
    else:
        if not Y_set.empty: # for non prediction use
            if Y_set.dtype == 'object':
                le = preprocessing.LabelEncoder()
                le.fit(Y_set)
                Y_encoder = le
                Y_set = le.transform(Y_set).ravel()
    #NaN to mean

    # TODO choose the strategy
    imp = preprocessing.Imputer(axis=0, verbose=1)
    imp = imp.fit(X_set)
    X_set = imp.transform(X_set)

    print('Pre processing results: X_set-{} Y_set-{}'.format(X_set.shape, Y_set.shape))
    return X_set, Y_set , X_encoders,Y_encoder