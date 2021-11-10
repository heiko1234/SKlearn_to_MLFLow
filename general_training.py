# Info


# Imports
from pathlib import Path
import os

import pandas as pd
import numpy as np
import yaml

import mlflow
import mlflow.sklearn

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

# from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold

# from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor

from dotenv import load_dotenv

load_dotenv()


pandas_dtypes = {
    "float64": "float",
    "int64": "integer",
    "bool": "boolean",
    "double": "double",
    "object": "string",
    "binary": "binary",
}


sklearn_models_dict = {
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "RandomForestRegressor": RandomForestRegressor(),
    "Lin_Reg": LinearRegression(),
    "Ridge": Ridge(),
    "AdaBoostRegressor": AdaBoostRegressor()
}




scaler_dict = {
    "MinMax": MinMaxScaler(),
    "Standard": StandardScaler(),
    "None": None,
}


def read_configuration(configuration_file_path):
    """This function reads the Residencetime_setup.yaml file
    from the Source Code Folder"
    Arg:
        configuration_file_path[str]: path to that file

    Returns:
        [type]: yaml configuration used in this pipeline script
    """

    with open(configuration_file_path) as file:
        configuration = yaml.full_load(file)

    return configuration


def replace_infs(df, with_value=np.nan):

    to_replace = [np.inf, -np.inf]

    return df.replace(to_replace, with_value)


def drop_nans(df: pd.DataFrame, axis=0):
    
    return df.dropna(axis)


def reset_index_train_test_split(
    feature_data, target_data, test_size=0.1, random_state=2021
):
    (
        features_train,
        features_test,
        target_train,
        target_test,
    ) = train_test_split(
        feature_data,
        target_data,
        test_size=test_size,
        random_state=random_state,
    )

    features_train = features_train.reset_index(drop=True)
    features_test = features_test.reset_index(drop=True)
    target_train = target_train.reset_index(drop=True)
    target_test = target_test.reset_index(drop=True)

    return features_train, features_test, target_train, target_test


def scale_data(train, test, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
    if len(train.shape) == 1 and len(test.shape) == 1:
        train = train.to_numpy()
        test = test.to_numpy()
        train_np = train.reshape(-1, 1)
        test_np = test.reshape(-1, 1)
    elif len(train.shape) == 2 and len(train.shape) == 2:
        train_np = train
        test_np = test
    else:
        print("different shapes of train and test")
    #
    train_np = scaler.fit_transform(train_np)
    test_np = scaler.transform(test_np)
    #
    return train_np, test_np, scaler


def training(sk_model, x_train, y_train, MLFlow=False):
    sk_model = sk_model.fit(x_train, y_train)
    train_score = round(sk_model.score(x_train, y_train), 4)
    print(f"Training Score: {train_score}")

    if MLFlow:
        mlflow.log_metric("train_score", train_score)


def evaluate(sk_model, x_test, y_test, x_train, y_train, MLFLow=False):
    eval_score = round(sk_model.score(x_test, y_test), 4)
    eval_score_train = round(sk_model.score(x_train, y_train), 4)
    eval_nmse = mean_squared_error(
        y_true=y_test, y_pred=sk_model.predict(x_test), squared=False
    )
    eval_r2 = r2_score(y_true=y_test, y_pred=sk_model.predict(x_test))

    cv = KFold(n_splits=4)
    single_scores = cross_val_score(
        sk_model, x_test, y_test, scoring="r2", cv=cv, n_jobs=-1
    )
    eval_kfold = single_scores.mean()

    print(f"Evaluation Score: {eval_score}")

    if MLFLow:
        mlflow.log_metric("eval_score_train", eval_score_train)
        mlflow.log_metric("eval_score", eval_score)
        mlflow.log_metric("eval_nmse", eval_nmse)
        mlflow.log_metric("eval_r2", eval_r2)
        mlflow.log_metric("eval_kfold", eval_kfold)


def make_model_pipeline(sk_model, sk_model2=None, scaler=None, parameter_model=None, parameter_model2=None):

    if sk_model2 != None:
        sk_model_out = sk_model
        sk_model_out.base_estimator=sk_model2

        sk_model_out.base_estimator.set_params(**parameter_model2)
        sk_model_out.set_params(**parameter_model)

        

    elif parameter_model != None:

        sk_model_out = sk_model
        sk_model_out.set_params(**parameter_model)
    
    else:
        sk_model_out = sk_model
    
    print(f"{sk_model_out} was created")

    if scaler is not None:
        return Pipeline(steps=[("scaler", scaler), ("model", sk_model_out)])

    else:
        return sk_model_out



def create_data_dict(data):

    feature_data_minmax = data.describe().loc[["min", "max"], :]

    return feature_data_minmax.to_dict()


def create_dict_of_modelparameter(model_parameter):

    output = {}

    for element in model_parameter:
        key_list = list(element.keys())
        for key in key_list:
            output[key] = element[key]

    return output


def create_dict_of_EnsembledModel(configuration, model):
    output = {}
    for model_dict in configuration[model]:
        for key in model_dict.keys():
            output[key] = create_dict_of_modelparameter(model_dict[key])
    return output


def flatten_list(list_list):
    flat_list = []

    for element in list_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def keys_in_list_of_dicts(list_of_dicts):
    output=[]
    for element in list_of_dicts:
        output.append(list(element.keys())[0])
    return output


def convert_list_to_dict(list_of_dicts):
    output={}
    keys = keys_in_list_of_dicts(list_of_dicts)
    for count, value in enumerate(keys):
        output[keys[count]]= list_of_dicts[count][keys[count]]
    return output



def main(data):

    output_models={}

    path = Path(__file__).parent
    configuration = read_configuration(
        configuration_file_path=os.path.join(path, "training_config.yaml")
    )

    #configuration = read_configuration(configuration_file_path="/home/heiko/Repos/SKlearn_to_MLFLow/training_config.yaml")
    # configuration

    MLFlow = configuration["MLFlow"]
    print(f"MLFlow: {MLFlow}")

    #MLFlow = False

    features = configuration["features"]
    target = configuration["target"]

    # remove na from feature and target data set
    data = data[features + [target]]

    data = replace_infs(data, with_value=np.nan)

    data = drop_nans(data, axis=0)
    data = data.reset_index(drop=True)

    target_data = data[target]
    feature_data = data[features]

    # Input Schema for MLFlow tracking
    if MLFlow:
        input_schema = Schema(
            [
                ColSpec(
                    pandas_dtypes[str(feature_data.dtypes[element])], element
                )
                for element in feature_data.columns
            ]
        )
        output_schema = Schema(
            [ColSpec(pandas_dtypes[str(target_data.dtypes)])]
        )
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        data_minmax_dict = create_data_dict(data)

    # split in train and test data
    (
        features_train,
        features_test,
        target_train,
        target_test,
    ) = reset_index_train_test_split(
        feature_data,
        target_data,
        test_size=configuration["test_split"]["test_size"],
        random_state=configuration["test_split"]["random_state"],
    )

    target_train = target_train.tolist()
    target_test = target_test.tolist()

    for model in configuration["Model"]:
        print(f"Used Model: {model} ")

        for scaler in configuration["Scaler"]:
            print(f"Used scaler: {scaler} ")

            try:

                model_parameter = configuration[model]
                model_parameter_dict = create_dict_of_modelparameter(
                    model_parameter
                )

            except KeyError or ValueError:
                model_parameter_dict = None

            model_pipe = make_model_pipeline(
                sk_model=sklearn_models_dict[model],
                scaler=scaler_dict[scaler],
                parameter_model=model_parameter_dict,
            )

            if MLFlow:

                mlflow.set_experiment(configuration["MLFlow_Experiment"])

                with mlflow.start_run():

                    print("Model run: ", mlflow.active_run().info.run_uuid)

                    print("Training and Evaluation for MLFlow started.")

                    training(
                        sk_model=model_pipe,
                        x_train=features_train,
                        y_train=target_train,
                        MLFlow=True,
                    )
                    evaluate(
                        sk_model=model_pipe,
                        x_test=features_test,
                        y_test=target_test,
                        x_train=features_train,
                        y_train=target_train,
                        MLFLow=True,
                    )

                    print("starting to track artifacts in MLFlow.")

                    mlflow.sklearn.log_model(
                        model_pipe, "model", signature=signature
                    )

                    mlflow.set_tag("model_type", model)
                    mlflow.set_tag("scaler", scaler)

                    mlflow.set_tag("target", configuration["target"])
                    mlflow.set_tag("features", configuration["features"])

                    mlflow.set_tag("model_parameters", model_parameter_dict)

                    mlflow.log_dict(
                        data_minmax_dict, "model/feature_limits.json"
                    )
                    mlflow.log_dict(
                        model_parameter_dict, "model/model_parameters.json"
                    )

                mlflow.end_run()

            else:
                model_pipe.fit(features_train, target_train)
                model_pipe.score(features_test, target_test)

                training(
                    sk_model=model_pipe,
                    x_train=features_train,
                    y_train=target_train,
                    MLFlow=False,
                )
                evaluate(
                    sk_model=model_pipe,
                    x_test=features_test,
                    y_test=target_test,
                    x_train=features_train,
                    y_train=target_train,
                    MLFLow=False,
                )
                output_models[model] = model_pipe

    for model in keys_in_list_of_dicts(list_of_dicts = configuration["EnsembledModel"]):

        print(f"Used Ensemble Model: {model}")

        dict_models= convert_list_to_dict(list_of_dicts = configuration["EnsembledModel_Parameter"])

        for scaler in configuration["Scaler"]:
            print(f"Used scaler: {scaler} ")

            try: 

                dict_enmodel = convert_list_to_dict(dict_models[model])
                keys_of_enmodel = keys_in_list_of_dicts(dict_models[model])


                make_model_pipeline(sk_model = sklearn_models_dict[keys_of_enmodel[0]], 
                                    sk_model2=sklearn_models_dict[keys_of_enmodel[1]], 
                                    scaler=scaler_dict[scaler], 
                                    parameter_model=create_dict_of_modelparameter(model_parameter = dict_enmodel[keys_of_enmodel[0]]), 
                                    parameter_model2=create_dict_of_modelparameter(model_parameter = dict_enmodel[keys_of_enmodel[1]])
                                    )

                if MLFlow:

                    mlflow.set_experiment(configuration["MLFlow_Experiment"])

                    with mlflow.start_run():

                        print("Model run: ", mlflow.active_run().info.run_uuid)

                        print("Training and Evaluation for MLFlow started.")

                        training(
                            sk_model=model_pipe,
                            x_train=features_train,
                            y_train=target_train,
                            MLFlow=True,
                        )
                        evaluate(
                            sk_model=model_pipe,
                            x_test=features_test,
                            y_test=target_test,
                            x_train=features_train,
                            y_train=target_train,
                            MLFLow=True,
                        )

                        print("starting to track artifacts in MLFlow.")

                        mlflow.sklearn.log_model(
                            model_pipe, "model", signature=signature
                        )

                        mlflow.set_tag("model_type", model)
                        mlflow.set_tag("scaler", scaler)

                        mlflow.set_tag("target", configuration["target"])
                        mlflow.set_tag("features", configuration["features"])

                        mlflow.set_tag("model_parameters", model_parameter_dict)

                        mlflow.log_dict(
                            data_minmax_dict, "model/feature_limits.json"
                        )
                        mlflow.log_dict(
                            model_parameter_dict, "model/model_parameters.json"
                        )

                    mlflow.end_run()

                else:
                    model_pipe.fit(features_train, target_train)
                    model_pipe.score(features_test, target_test)

                    training(
                        sk_model=model_pipe,
                        x_train=features_train,
                        y_train=target_train,
                        MLFlow=False,
                    )
                    evaluate(
                        sk_model=model_pipe,
                        x_test=features_test,
                        y_test=target_test,
                        x_train=features_train,
                        y_train=target_train,
                        MLFLow=False,
                    )
                    output_models[model] = model_pipe
            
            except NameError or ValueError:
                pass
    
    if not MLFlow:
        return output_models





if __name__ == "__main__":

    load_dotenv()

    local_run = os.getenv("LOCAL_RUN", False)

    if local_run:
        # path_data = "/home/heiko/Repos/SKlearn_to_MLFLow/data/Filtered_Data2.parquet"

        path = Path(__file__).parent
        path_data = os.path.join(path, "data/Filtered_Data2.parquet")

    else: 

        path_dat = os.GETENV("DATA_PATH", "/data/Filtered_Data2.parquet")

    data = pd.read_parquet(
        path_data
    )

    main(data)



