

# from os import getenv
from dotenv import load_dotenv

from pathlib import PurePosixPath
from pathlib import Path
import pickle
import mlflow

import pandas as pd
import numpy as np

# import logging
import os
import json
import copy

from dash import dcc as dcc


load_dotenv()

local_run = os.getenv("LOCAL_RUN", False)
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("BLOB_MODEL_CONTAINER_NAME")



mlflow_dtypes = {
    "float": "float32",
    "integer": "int32",
    "boolean": "bool",
    "double": "double",
    "string": "object",
    "binary": "binary",
}



def get_mlflow_model(model_name, azure=True, model_dir = "/model/"):

    if azure:
        model_dir = os.getenv("MLFLOW_MODEL_DIRECTORY", "models:/")
        model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Staging")
        artifact_path = PurePosixPath(model_dir).joinpath(model_name, model_stage)
        artifact_path

        model = mlflow.pyfunc.load_model(str(artifact_path))
        print(f"Model {model_name} loaden from Azure: {artifact_path}")

    if not azure:
        model = pickle.load(open(f"{model_dir}/{model_name}/model.pkl", 'rb'))
        print(f"Model {model_name} loaded from local pickle file")

    return model


def get_model_json_artifact(
    local=True,
    path=None,
    model_path="models",
    model_name=None,
    features="feature_dtypes.json",
):
    """This function loads json file form a dumped mlflow model

    Args:
        local (bool, optional): [description]. Defaults to True.
        path ([type], optional): in docker: "/model/", else folder where models are saved.
        model_path (str, optional): [description]. Defaults to "models".
        model_name ([type], optional): [sklearn model name]. Defaults to None.
        features (str, optional): feature_dtypes.json/ feature_limits.json

    Returns:
        [type]: [json file]
    """

    if local:
        if path is None:
            path = Path(__file___).parent
            # print(f"Parentspath: {path}")
    if not local:
        # Access the artifacts to "/model/model_name/file" for the docker.
        path = "/model/"
        model_path = ""

    path_load = os.path.join(path, model_path, model_name, features)

    return json.loads(open(path_load, "r").read())


def create_all_model_json_dict(local=True,
    path=None,
    model_path="models",
    features="feature_dtypes.json"):
    output = {}
    folderpath = os.path.join(path, model_path)
    for folder in os.listdir(folderpath):
        if os.path.isdir(os.path.join(folderpath, folder)):
            output[folder] = get_model_json_artifact(
                            local=local,
                            path=path,
                            model_path=model_path,
                            model_name=folder,
                            features=features,
                        )
    return output


def flatten_dict(nested_dict):
    output={}
    for key in nested_dict.keys():
        for second_key in nested_dict[key].keys():
            if second_key not in output:
                output[second_key] = nested_dict[key][second_key]
    return output


def flatten_consolidate_dict(nested_dict, take_lower_min=True, take_higher_max=True):
    output={}
    for key in nested_dict.keys():
        for second_key in nested_dict[key].keys():
            if second_key not in output:
                output[second_key] = copy.deepcopy(nested_dict[key][second_key])
            if second_key in output:
                if take_lower_min:
                    if output[second_key]["min"] > nested_dict[key][second_key]["min"]:
                        output[second_key]["min"] = copy.deepcopy(nested_dict[key][second_key]["min"])
                else:
                    if output[second_key]["min"] < nested_dict[key][second_key]["min"]:
                        output[second_key]["min"] = copy.deepcopy(nested_dict[key][second_key]["min"])
                if take_higher_max:
                    if output[second_key]["max"] < nested_dict[key][second_key]["max"]:
                        output[second_key]["max"] = copy.deepcopy(nested_dict[key][second_key]["max"])
                else:
                    if output[second_key]["max"] > nested_dict[key][second_key]["max"]:
                        output[second_key]["max"] = copy.deepcopy(nested_dict[key][second_key]["max"])
    return output


def create_warning(TAG_limit_dict, key, value, digits=2):
    if key in TAG_limit_dict.keys():
        if value < TAG_limit_dict[key]["min"]:
            return dcc.Markdown(
                f"""{key} is below min value of model:
                        {round(TAG_limit_dict[key]["min"], digits)} """
            )

        elif value > TAG_limit_dict[key]["max"]:
            return dcc.Markdown(
                f"""{key} is above max value of model:
                        {round(TAG_limit_dict[key]["max"], digits)} """
            )

        else:
            return None

    else:
        return None


def transform_df_to_mlflow_df(data, dtype_dict, mlflow_dtypes):
    for element in list(dtype_dict.keys()):
        try:
            data[element] = data[element].astype(
                mlflow_dtypes[dtype_dict[element]]
            )
        except BaseException:
            continue
    return data




# Feature Limit Dict 

feature_limits_dict = create_all_model_json_dict(local=True,
    path="/home/heiko/Repos/SKlearn_to_MLFLow",
    model_path="model_dump",
    features="feature_limits.json")
feature_limits_dict


flat_featue_limits_dict= flatten_dict(nested_dict=feature_limits_dict)
flat_featue_limits_dict


flatten_consolidate_dict(nested_dict = feature_limits_dict, take_lower_min=True, take_higher_max=True)
flatten_consolidate_dict(nested_dict = feature_limits_dict, take_lower_min=True, take_higher_max=False)
flatten_consolidate_dict(nested_dict = feature_limits_dict, take_lower_min=False, take_higher_max=False)
flatten_consolidate_dict(nested_dict = feature_limits_dict, take_lower_min=False, take_higher_max=True)


TAG_limit_dict = flatten_consolidate_dict(nested_dict = feature_limits_dict, take_lower_min=True, take_higher_max=True)


create_warning(TAG_limit_dict=TAG_limit_dict, key = "ManufacturingProcess42", value=200) #yes
create_warning(TAG_limit_dict=TAG_limit_dict, key = "ManufacturingProcess42", value=160) #no
create_warning(TAG_limit_dict=TAG_limit_dict, key = "ManufacturingProcess42", value=120) #yes



# Feature dtype Dict


get_model_json_artifact(
    local=True,
    path="/home/heiko/Repos/SKlearn_to_MLFLow",
    model_path="model_dump",
    model_name="dashapp_model",
    features="feature_dtypes.json",
)


feature_dtypes_dict = create_all_model_json_dict(local=True,
    path="/home/heiko/Repos/SKlearn_to_MLFLow",
    model_path="model_dump",
    features="feature_dtypes.json")
feature_dtypes_dict


dtype_dict=flatten_dict(nested_dict=feature_dtypes_dict)
dtype_dict




# Create a pd.DataFrame

MP09 = 40  #38.89..49.36
MP32 = 160  #150..170
MP13 = 36 # 32.1..38.6
BM02 = 60  # 51.28..64.75
MP20 = 4600  # 4392..4759
MP22 = 6  # 1..12
MP42 = 140  #130..190 (secondmodel feature_limits.json)


data = pd.DataFrame(
    data=[[MP09, MP32, MP13, BM02, MP20, MP22]],
    columns=[
        "ManufacturingProcess09",
        "ManufacturingProcess32",
        "ManufacturingProcess13",
        "BiologicalMaterial02",
        "ManufacturingProcess20",
        "ManufacturingProcess22",
    ],
)
data

# same parameter but different order
data2 = pd.DataFrame(
    data=[[MP09, MP13, MP32, MP20, MP22, BM02]],
    columns=[
        "ManufacturingProcess09",
        "ManufacturingProcess13",
        "ManufacturingProcess32",
        "ManufacturingProcess20",
        "ManufacturingProcess22",
        "BiologicalMaterial02",
    ],
)
data2

# one extra parameter
data3 = pd.DataFrame(
    data=[[MP09, MP32, MP13, MP42, BM02, MP20, MP22]],
    columns=[
        "ManufacturingProcess09",
        "ManufacturingProcess32",
        "ManufacturingProcess13",
        "ManufacturingProcess42",
        "BiologicalMaterial02",
        "ManufacturingProcess20",
        "ManufacturingProcess22",
    ],
)
data3


# Change data dtypes

data = transform_df_to_mlflow_df(data = data, dtype_dict=dtype_dict, mlflow_dtypes=mlflow_dtypes)
data2 = transform_df_to_mlflow_df(data = data2, dtype_dict=dtype_dict, mlflow_dtypes=mlflow_dtypes)
data3 = transform_df_to_mlflow_df(data = data3, dtype_dict=dtype_dict, mlflow_dtypes=mlflow_dtypes)


# Load a MLFlow Model either from local artifact or from MLFlow Docker container

# local pickle file
model= get_mlflow_model(model_name="dashapp_model", azure=False, model_dir = "/home/heiko/Repos/SKlearn_to_MLFLow/model_dump")

model.predict(data)  #39.51
model.predict(data2) #37.0
model.predict(data3)  # error: 7 features instead 6


# azure mlflow.pyfunc.load_model()
model= get_mlflow_model(model_name="dashapp_model", azure=True, model_dir = None)

model.predict(data)  #39.51
model.predict(data2)  #39.51
model.predict(data3)  #39.51
