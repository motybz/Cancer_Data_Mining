from shared_parameters import *
import numpy as np
import glob

MODELS_LOCATION = 'C:/Users/motibz/Documents/Studing/'#LR_7_features.sav'
def get_row_len(row):
    length  = 0
    for key, val in row.items():
        if val:
           length+=1
    return length
def choose_model_file(models_location,row):
    num_of_fetures = get_row_len(row)
    for model_file in glob.glob(os.path.join(models_location,'*.sav')):
        model_num_cahr = model_file.split('_')[1]
        try:
            model_num = int(model_num_cahr)
        except:
            print("Wrong file name convention")
            continue
        if model_num == num_of_fetures:
            return get_trained_model(os.path.join(models_location,model_file))
    print("No sutible model")
    return None



def get_trained_model(path_to_file):
    # load the model from disk
    loaded_model = pickle.load(open(path_to_file, 'rb'))
    return loaded_model
    #result = loaded_model.score(X_test, Y_test)

def row_dict_to_DataSet(row,dataset):
    print('Comparing suorce model {} to given row {}'.format(dataset.feature_list,row.keys()))
    X = pd.DataFrame(index=[1], columns=dataset.feature_list)
    Y = pd.DataFrame(index=[1]) #empty one
    feature_list = []
    for i,feature in enumerate(dataset.feature_list):
        value = row.get(feature)
        if value:
            X.set_value(1,feature,value)
            feature_list.append(feature)
        else:
            # TODO see if NaN avalible
            # X.set_value(1, feature,np.NaN )
            print('{} value not found in the given row'.format(feature))
            return None
    modified_X_set, Y_set, X_encoders,Y_encoder = pre_processing(X, Y,dataset.X_encoders)
    return DataSet(modified_X_set,Y,feature_list,X_encoders,dataset.Y_encoder)

def _uppercase_for_dict_keys(lower_dict):
    upper_dict = {}
    for k, v in lower_dict.items():
        if isinstance(v, dict):
            v = _uppercase_for_dict_keys(v)
        upper_dict[k.upper()] = v
    return upper_dict

def get_prediction(row):
    trained_model = choose_model_file(MODELS_LOCATION,row)
    dict_encoder_prediction = {}
    if trained_model:
        upper_row = _uppercase_for_dict_keys(row)
        user_row_dataset = row_dict_to_DataSet(upper_row,trained_model.train_set)
        if user_row_dataset:
            model = trained_model.fited_model
            prediction = model.predict_proba(user_row_dataset.X)
            classes = model.classes_
            classes_names = trained_model.train_set.Y_encoder.inverse_transform(classes)
            for i,value in enumerate(prediction[0]):
                dict_encoder_prediction.update({classes_names[i]:"{:.1%}".format(value)})
            model_score = trained_model.train_score
            dict_encoder_prediction.update({'Accuracy Level': "{:.1%}".format(model_score)})
    else:
        dict_encoder_prediction.update({'Cant find suitable model':None})

    return dict_encoder_prediction