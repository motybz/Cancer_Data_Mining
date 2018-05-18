from shared_parameters import *
import numpy as np

MODEL_LOCATION = 'C:/Users/motibz/Documents/Studing/LR_7_features.sav'
def get_trained_model(path_to_file):
    # load the model from disk
    loaded_model = pickle.load(open(path_to_file, 'rb'))
    return loaded_model
    #result = loaded_model.score(X_test, Y_test)

def row_dict_to_DataSet(row,dataset):
    X = pd.DataFrame(index=[1], columns=dataset.feature_list)
    Y = pd.DataFrame(index=[1]) #empty one
    feature_list = []
    for i,feature in enumerate(dataset.feature_list):
        value = row.get(feature)
        if value:
            X.set_value(1,feature,value)
            feature_list.append(feature)
        else:
            X.set_value(1, feature,np.NaN )
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
    trained_model = get_trained_model(MODEL_LOCATION)
    upper_row = _uppercase_for_dict_keys(row)
    user_row_dataset = row_dict_to_DataSet(upper_row,trained_model.train_set)
    model = trained_model.fited_model
    prediction = model.predict_proba(user_row_dataset.X)
    dict_encoder_prediction = {}
    classes = model.classes_
    classes_names = trained_model.train_set.Y_encoder.inverse_transform(classes)
    for i,value in enumerate(prediction[0]):
        dict_encoder_prediction.update({classes_names[i]:"{:.1%}".format(value)})

    return dict_encoder_prediction