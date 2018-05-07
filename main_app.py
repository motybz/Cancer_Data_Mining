import os ,yaml,sys
# import pandas as pd
from operator import itemgetter
from tools.import_data import *
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import feature_selection as fs

# consts
FILE_PATH = os.path.dirname(os.path.abspath("__file__"))
CONFIG_FILE = os.path.join(FILE_PATH, 'config.yml')
config = yaml.load(open(CONFIG_FILE, 'r'))
TRAIN_FILE = config['files']['train_set']
TEST_FILE = config['files']['test_set']
SCORING = config['score']

models = []

models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))

# Classes
class DataSet:
    def __init__(self, X, Y, feature_list,encoding_dict, threshold=None, ratio=None):
        self.X = X
        self.Y = Y
        self.feature_list = feature_list
        self.encoding = encoding_dict # {feature_name: encoder}
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

    def test_the_model(self):
        self.test_score = test_the_data(self.test_set.X, self.test_set.Y, self.fited_model)
        return self.test_score

    # def set_chosen_model(self, name, fited_model):
    #     self.chosen_model = SkFModel(name, fited_model)

    def train_model(self):
        clf = self.chosen_model.skmodel
        clf.fit(self.train_set.X, self.train_set.Y)
        self.fited_model = clf



def pre_processing(X_set, Y_set ,encoders=None):
    # Data pre processing
    # Encoding Categorial features and imputing NaN's
    # https://chrisalbon.com/machine_learning/preprocessing_structured_data/convert_pandas_categorical_column_into_integers_for_scikit-learn/
    # http://pbpython.com/categorical-encoding.html
    # https://datascience.stackexchange.com/questions/14069/mass-convert-categorical-columns-in-pandas-not-one-hot-encoding

    # String categories to int
    if encoders:
        for feature, le in encoders.items():
            if feature in X_set.columns:
                X_set[feature][pd.isnull(X_set[feature])] = 'NaN'
                X_set[feature] = le.transform(X_set[feature])
            elif Y_set.name == feature:
                Y_set = le.transform(Y_set)
    else:
        char_cols = X_set.dtypes.pipe(lambda x: x[x == 'object']).index
        encoders = {}
        for feature in char_cols:
            # https://stackoverflow.com/questions/36808434/label-encoder-encoding-missing-values
            X_set[feature][pd.isnull(X_set[feature])] = 'NaN'
            le = preprocessing.LabelEncoder()
            le.fit(X_set[feature])
            X_set[feature] = le.transform(X_set[feature])
            encoders.update({feature:le})
        # Also for Y set
        if Y_set.dtype == 'object':
            le = preprocessing.LabelEncoder()
            le.fit(Y_set)
            encoders.update({Y_set.name: le})
            Y_set = le.transform(Y_set).ravel()
    #NaN to mean

    # TODO choose the strategy
    imp = preprocessing.Imputer(axis=0, verbose=1)
    imp = imp.fit(X_set)
    X_set = imp.transform(X_set)

    print('Pre processing results: X_set-{} Y_set-{}'.format(X_set.shape, Y_set.shape))
    return X_set, Y_set , encoders

def get_models_CV_scores(X_train, Y_train, models):
    # Spot Check Algorithms with cross validation
    # evaluate each model in turn
    scores = []
    names = []
    results = []
    for name, model in models:
        kfold = model_selection.StratifiedKFold(n_splits=10, shuffle=True)
        try:
            cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, n_jobs=-1, scoring=SCORING)
        except ValueError as e:
            print('ValueError{}'.format(None))
        else:
            scores.append(cv_results)
            names.append(name)
            results.append({"model": SkFModel(name, model), "score": cv_results.mean()})
    # Compare Algorithms
    # fig = plt.figure()
    # fig.suptitle('Algorithm Comparison')
    # ax = fig.add_subplot(111)
    # plt.boxplot(scores)
    # ax.set_xticklabels(names)
    # plt.show()
    return results

def test_the_data(X_test, Y_test, fited_model):
    # Make predictions on test dataset
    scores = []
    scorer = get_scorer(SCORING)
    score = scorer(fited_model,X_test, Y_test)
    scores.append(score)
    return scores

def get_the_best(results):  # input - list of dict {"name":name,"score":score}
    m = max([k["score"] for k in results])
    i = [k["score"] for k in results].index(m)
    model = results[i]["model"]
    print('The best model for the given data is: {} with the score {}'.format(model.name, m))
    # print ('The best model for the given train (' + X_train.shape +') is: ' + name + 'with the score ' +m )
    return results[i]  # the max model (dict type)

if __name__ == "__main__":
    data_sets = []
    X_test = None
    X_train, Y_train, original_headers_train = load_dataset(TRAIN_FILE)
    data_sets.append({'X_train': X_train, 'Y_train': Y_train, 'original_headers_train': original_headers_train})
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', Y_train.shape)
    X_train, Y_train ,encoders = pre_processing(X_train, Y_train)
    if TEST_FILE:
        X_test, Y_test, original_headers_test = load_dataset(TEST_FILE)
        data_sets.append({'X_test': X_test, 'Y_test': Y_test, 'original_headers_test': original_headers_test})
        print('Test data shape: ', X_test.shape)
        print('Test labels shape: ', Y_test.shape)
        X_test, Y_test, encoders = pre_processing(X_test, Y_test ,encoders)

    #create feature score list
    fs_scores = fs.mutual_info_classif(X_train,Y_train)
    feature_scores_list = []
    for i,score in enumerate(fs_scores):
        feature_scores_list.append({'feature_name':original_headers_train[i],'score':score})
    feature_scores_list.sort(key=itemgetter('score'),reverse=True)
    for feature in feature_scores_list:
        print (feature)

    # create the new datasets according to the feature selection ratio
    new_trains = []
    new_tests = []
    myrange = np.arange(0.01, 1, 0.01)
    for VTHRESH in myrange:
        sel = fs.SelectPercentile(score_func=fs.mutual_info_classif, percentile=VTHRESH * 100)
        sel.fit(X_train, Y_train)
        X_train_mod = sel.transform(X_train)
        if len(new_trains) >= 1:
            if new_trains[-1].ratio == X_train_mod.shape[-1] / X_train.shape[-1]:
                continue
        mask = sel.get_support()  # list of booleans
        sliced_features = []  # The list of the sliced K best features
        sliced_encoders = {}  # The list of the sliced encoders
        for ans, feature in zip(mask, original_headers_train):
            if ans:
                sliced_features.append(feature)
                if feature in encoders.keys():
                    sliced_encoders.update({feature:encoders[feature]})
        new_trains.append(DataSet(X_train_mod,Y_train,sliced_features,sliced_encoders, VTHRESH,
                                  X_train_mod.shape[-1] / X_train.shape[-1],)
                              )
        if TEST_FILE:
            sliced_features = []  # The list of the sliced K best features
            sliced_encoders = {}  # The list of the sliced encoders
            for ans, feature in zip(mask, original_headers_test):
                if ans:
                    sliced_features.append(feature)
                    if feature in encoders.keys():
                        sliced_encoders.update({feature: encoders[feature]})
            X_test_mod = sel.transform(X_test)
            new_tests.append(DataSet(X_test_mod,Y_test,sliced_features,sliced_encoders, VTHRESH,
                                     X_test_mod.shape[-1] / X_test.shape[-1],)
                              )

    best_traind_models = []
    new_models = []
    for i, train_set in enumerate(new_trains):
        print("**Training Section** for: {} features ({})".format(str(train_set.X.shape[-1]),train_set.feature_list))
        train_results = get_models_CV_scores(train_set.X, train_set.Y, models)
        new_models.append({'train_set': train_set, 'train_resultes': train_results})
        if train_results:
            best_tr = get_the_best(train_results)
        best_traind_models.append(TrainedModel(train_set,best_tr['model'],best_tr['score']))
        if new_tests:
            best_traind_models[-1].test_set = new_tests[i]


        # best_tr_results.append({"model":best_tr['model'],
        #                         "score":best_tr['score'],
        #                         "num_fetures":int(train_set.X.shape[-1]),
        #                         'features_names':train_set['features_name']})

        for model in train_results:
            print(model['model'].name, model['score'])

    plt.plot([d.ratio for d in new_trains],
             [d.train_score for d in best_traind_models])
    plt.legend()
    plt.show()

    sorted_reasultes = sorted(best_traind_models,key=lambda x: x.train_score)
    for best in sorted_reasultes:
        print("The best model for {} fetures is {} with teht score: {} and the feturaes {}".format(len(best.train_set.feature_list),
                                                                                                   best.chosen_model.name,
                                                                                                   best.train_score,
                                                                                                   best.train_set.feature_list))
    # need to modify the test shape
    print("**Testing Section:**")
    for trained_model in best_traind_models:
        print('Testing {} on {} with the model {}'.format(trained_model.train_set.X.shape,
                                                          trained_model.test_set.X.shape,
                                                          trained_model.chosen_model.name)
              )
        trained_model.train_model()
        trained_model.test_the_model()
        print ('Score - {}'.format(trained_model.test_score))