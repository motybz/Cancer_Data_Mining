import os ,yaml,sys
sys.path.append('C:/Users/motibz/Documents/Studing/Cancer_Data_Mining')
from operator import itemgetter
from tools.import_data import *
import matplotlib.pyplot as plt
from sklearn import model_selection
from shared_parameters import *
from sklearn import feature_selection as fs

# consts
FILE_PATH = os.path.dirname(os.path.abspath("__file__"))
CONFIG_FILE = os.path.join(FILE_PATH, 'config.yml')
config = yaml.load(open(CONFIG_FILE, 'r'))
TRAIN_FILE = config['files']['train_set']
TEST_FILE = config['files']['test_set']
SCORING = config['score']
OUTPUT = config['outputs']['outdir']
CV_LOOPS = config['cv_loops']
if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)
if config['outputs']['save_stdout']:
    sys.stdout = open(os.path.join(OUTPUT,'{}_stdout.txt'.format(os.path.basename(TRAIN_FILE))), 'a+')
# TODO get model by name
models = []

# models.append(('MLP', MLPClassifier()))
models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))



def get_models_CV_scores(X_train, Y_train, models,cv_loops):
    # Spot Check Algorithms with cross validation
    # evaluate each model in turn
    scores = []
    names = []
    results = []
    for name, model in models:
        cv_results_colection = None
        kfold = model_selection.RepeatedStratifiedKFold(n_repeats=cv_loops,n_splits=10)
        try:
            cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, n_jobs=-1, scoring=SCORING)
        except ValueError as e:
            print('ValueError{}'.format(None))
        else:
            # cv_results_colection.extend(cv_results)
            scores.append(cv_results)
            names.append(name)
            results.append({"model": SkFModel(name, model), "score": cv_results.mean()})
    # Compare Algorithms
    fig = plt.figure()
    plt.title('Source-{} ,{} Attributes'.format(os.path.basename(TRAIN_FILE),X_train.shape[-1]), fontsize=10)
    plt.suptitle('Algorithm Comparison',fontsize=16)
    ax = fig.add_subplot(111)
    plt.boxplot(scores)
    ax.set_xticklabels(names)
    plt.ylabel('Score ({})'.format(SCORING), fontsize=14)
    if config['outputs']['save_charts']:
        fig.savefig(os.path.join(OUTPUT,'{}_Algorithm_Comparison.png'.format(X_train.shape[-1])), dpi=1200)
    if config['outputs']['show_charts']:
        plt.show()
    plt.close(fig)
    return results

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
    X_train, Y_train ,X_encoders,Y_encoder = pre_processing(X_train, Y_train)
    if TEST_FILE:
        X_test, Y_test, original_headers_test = load_dataset(TEST_FILE)
        data_sets.append({'X_test': X_test, 'Y_test': Y_test, 'original_headers_test': original_headers_test})
        print('Test data shape: ', X_test.shape)
        print('Test labels shape: ', Y_test.shape)
        X_test, Y_test,X_encoders,Y_encoder = pre_processing(X_test, Y_test ,X_encoders,Y_encoder)

    #create feature score list
    #TODO use the same function and rmove zeros
    # fs_scores = fs.mutual_info_classif(X_train,Y_train)
    # feature_scores_list = []
    # for i,score in enumerate(fs_scores):
    #     feature_scores_list.append({'feature_name':original_headers_train[i],'score':score})
    # feature_scores_list.sort(key=itemgetter('score'),reverse=True)
    # for feature in feature_scores_list:
    #     print (feature)

    # create the new datasets according to the feature selection ratio
    new_trains = []
    new_tests = []
    # igain = information_gain(X_train, Y_train)
    sel = fs.SelectPercentile(score_func=fs.f_classif)
    sel.fit(X_train, Y_train)
    feature_scores_list = []
    for i,score in enumerate(sel.scores_):
        feature_scores_list.append({'feature_name':original_headers_train[i],'score':score})
    feature_scores_list.sort(key=itemgetter('score'),reverse=True)
    for feature in feature_scores_list:
        print (feature)
    myrange = np.arange(0.01, 1, 0.01)
    # myrange.[::-1].sort()
    for f_score_threshold in myrange:
        sel.set_params(percentile=f_score_threshold * 100)
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
                if feature in X_encoders.keys():
                    sliced_encoders.update({feature:X_encoders[feature]})
        new_trains.append(DataSet(X_train_mod,Y_train,sliced_features,sliced_encoders,Y_encoder, f_score_threshold,
                                  X_train_mod.shape[-1] / X_train.shape[-1],)
                              )
        if TEST_FILE:
            sliced_features = []  # The list of the sliced K best features
            sliced_encoders = {}  # The list of the sliced encoders
            for ans, feature in zip(mask, original_headers_test):
                if ans:
                    sliced_features.append(feature)
                    if feature in X_encoders.keys():
                        sliced_encoders.update({feature: X_encoders[feature]})
            X_test_mod = sel.transform(X_test)
            new_tests.append(DataSet(X_test_mod,Y_test,sliced_features,sliced_encoders,Y_encoder, f_score_threshold,
                                     X_test_mod.shape[-1] / X_test.shape[-1],)
                              )

    best_traind_models = []
    new_models = []
    for i, train_set in enumerate(new_trains):
        print("**Training Section** for: {} features ({})".format(str(train_set.X.shape[-1]),train_set.feature_list))
        train_results = get_models_CV_scores(train_set.X, train_set.Y, models,CV_LOOPS)
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
    newfig = plt.figure()
    plt.plot([d.ratio for d in new_trains],
             [d.train_score for d in best_traind_models],'--bo')
    plt.legend()
    plt.xlabel('Percentage of attributes used', fontsize=14)
    plt.ylabel('Score ({})'.format(SCORING), fontsize=14)
    if config['outputs']['save_charts']:
        newfig.savefig(os.path.join(OUTPUT, 'Features_VS_Score.png'),dpi=1200)
    if config['outputs']['show_charts']:
        plt.show()
    plt.close(newfig)
    sorted_reasultes = sorted(best_traind_models,key=lambda x: x.train_score)
    for best in sorted_reasultes:
        print("The best model for {} fetures is {} with the score: {} and the feturaes {}".format(len(best.train_set.feature_list),
                                                                                                   best.chosen_model.name,
                                                                                                   best.train_score,
                                                                                                   best.train_set.feature_list))
    for trained_model in best_traind_models:
        trained_model.train_model()

    if TEST_FILE:
        print("**Testing Section:**")
        for trained_model in best_traind_models:
            trained_model.test_the_model(SCORING)

        sorted_reasultes = sorted(best_traind_models, key=lambda x: x.test_score)
        for reasulte in sorted_reasultes:
            print('Testing {} on {} with the model {}'.format(reasulte.train_set.X.shape,
                                                              reasulte.test_set.X.shape,
                                                              reasulte.chosen_model.name))
            print('Score - {}'.format(reasulte.test_score))

    #save trained model to files
    #https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    if config['outputs']['save_models_agenda'] == 'all':
        for model in sorted_reasultes:
            model.save_traind_model_to_file(OUTPUT)
    elif config['outputs']['save_models_agenda'] == 'best':
        sorted_reasultes[-1].save_traind_model_to_file(OUTPUT)