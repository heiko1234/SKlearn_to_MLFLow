

# from os import getenv
from dotenv import load_dotenv

from pathlib import PurePosixPath
from pathlib import Path
import pickle
import mlflow

import pandas as pd
# import numpy as np

# import logging
import os
import json
import copy
import mlflow
from pathlib import Path, PurePosixPath
import pickle

from azure.storage.blob import BlobServiceClient


from dash import dcc as dcc


load_dotenv()

local_run = os.getenv("LOCAL_RUN", False)
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("BLOB_MODEL_CONTAINER_NAME")
container_name


def get_mlflow_model(model_name, azure=True, local_model_dir = "/model/"):

    if azure:
        azure_model_dir = os.getenv("MLFLOW_MODEL_DIRECTORY", "models:/")
        model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Staging")
        artifact_path = PurePosixPath(azure_model_dir).joinpath(model_name, model_stage)
        artifact_path

        model = mlflow.pyfunc.load_model(str(artifact_path))
        print(f"Model {model_name} loaden from Azure: {artifact_path}")

    if not azure:
        model = pickle.load(open(f"{local_model_dir}/{model_name}/model.pkl", 'rb'))
        print(f"Model {model_name} loaded from local pickle file")

    return model


def read_model_json_from_blob(connection_string, container_name, model_name, filename):
    # get mlflow model directory in blob: "models:/""
    model_dir = os.getenv("MLFLOW_MODEL_DIRECTORY", "models:")
    # get stage: "Staging"
    model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Staging")
    # get artifact path of mode with model_name on Stage: "Staging"
    artifact_path = PurePosixPath(model_dir).joinpath(model_name, model_stage)
    # load that model
    model = mlflow.pyfunc.load_model(str(artifact_path))
    # get the loaded model runid
    model_id=model.metadata.run_id
    
    client = BlobServiceClient.from_connection_string(
        connection_string
    )
    # container blob client to container of mlflow
    container_client = client.get_container_client(container_name)

    # create file client for blob with a specific filename, of staged model

    for blob in container_client.list_blobs():
        if model_id in blob.name and filename in blob.name:
            # print(blob.name)

            f_client = client.get_blob_client(
                container=container_name, blob=blob.name
            )
    
            tempfile = os.path.join("temp.json")
            # dir_to_create = "".join(tempfile.split("/")[0:-1])
            # make folder path if it does not exist
            # os.makedirs(dir_to_create, exist_ok=True)

            with open(tempfile, "wb") as file:
                blob_data = f_client.download_blob()
                blob_data.readinto(file)

            try: 
                return json.loads(open(tempfile, "r").read())
            # except BaseException:
            #    print(f"seem to be no file: {filename} in blob: {container_name} available")
            finally:
                # finally remove temporary file
                Path(tempfile).unlink()


def get_model_json_artifact(
    azure=True,
    path=None,
    model_name=None,
    features="feature_dtypes.json",
):
    """This function loads json file form a dumped mlflow model or
    temporary dumps it to load it directly from azure / azurite

    Args:
        azure (bool, optional): [description]. Defaults to True.
        path ([type], optional): in docker: "/model/", else folder where models are saved.
        model_path (str, optional): [description]. Defaults to "models".
        model_name ([type], optional): [sklearn model name]. Defaults to None.
        features (str, optional): feature_dtypes.json/ feature_limits.json

    Returns:
        [type]: [json file]
    """

    if not azure:
        # Access the artifacts to "/model/model_name/file" for the docker.

        path_load = os.path.join(path, model_name, features)

        return json.loads(open(path_load, "r").read())
    
    if azure:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name = os.getenv("BLOB_MODEL_CONTAINER_NAME")

        file = read_model_json_from_blob(connection_string=connection_string, 
                        container_name=container_name, 
                        model_name=model_name, 
                        filename=features)
        if file: 
            return file
        else: 
            print(f"Warning: seem to be no file: {features} in blob: {container_name} available")


def create_all_model_json_dict(local=True,
    path=None,
    model_path=None,
    features="feature_dtypes.json",
    list_of_models=None):

    output = {}
    if local: 
        if model_path:
            folderpath = os.path.join(path, model_path)
        else:
            folderpath = path
        for folder in os.listdir(folderpath):
            if os.path.isdir(os.path.join(folderpath, folder)):
                output[folder] = get_model_json_artifact(
                                azure=False,
                                path=folderpath,
                                model_name=folder,
                                features=features,
                            )

    if not local and list_of_models:
        for modelname in list_of_models:
            output[modelname]=get_model_json_artifact(
                                    azure=True,
                                    path=None,
                                    model_name=modelname,
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


def decode_df_mlflow_dtype(data, dtype_dict):

    mlflow_dtypes = {
        "float": "float32",
        "integer": "int32",
        "boolean": "bool",
        "double": "double",
        "string": "object",
        "binary": "binary",
    }

    for element in list(dtype_dict.keys()):
        try:
            data[element] = data[element].astype(
                mlflow_dtypes[dtype_dict[element]]
            )
        except BaseException:
            continue
    return data


#####
####



MFI_dtypes =get_model_json_artifact(
            azure=True,
            path=None,
            model_name="MFI_polymer",
            features="feature_dtypes.json",
        )

MFI_dtypes

MFI_limits =get_model_json_artifact(
            azure=True,
            path=None,
            model_name="MFI_polymer",
            features="feature_limits.json",
        )
MFI_limits


limits = read_model_json_from_blob(connection_string=connection_string, 
                container_name=container_name, 
                model_name = "CI_polymer", 
                filename="feature_limits.json")
limits


# Feature Limit Dict 

feature_limits_dict = create_all_model_json_dict(local=True,
    path="/home/heiko/Repos/SKlearn_to_MLFLow/model_dump",
    model_path=None,
    features="feature_limits.json")
feature_limits_dict



feature_limits_dict = create_all_model_json_dict(local=False,
    path=None,
    model_path=None,
    features="feature_limits.json",
    list_of_models=["MFI_polymer", "CI_polymer"])
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

# if very thing is locally dumped
get_model_json_artifact(
    azure=False,
    path="/home/heiko/Repos/SKlearn_to_MLFLow/model_dump",
    model_name="dashapp_model",
    features="feature_dtypes.json",
)


feature_dtypes_dict = create_all_model_json_dict(local=True,
    path="/home/heiko/Repos/SKlearn_to_MLFLow/model_dump",
    features="feature_dtypes.json")
feature_dtypes_dict


# if every thing is n mlflow server, azurite
model2_featues=get_model_json_artifact(
                    azure=True,
                    path=None,
                    model_name="dashapp_model2",
                    features="feature_limits.json",
                )
model2_featues


model2_dtypes=get_model_json_artifact(
                    azure=True,
                    path=None,
                    model_name="dashapp_model2",
                    features="feature_dtype.json",
                )
model2_dtypes


feature_dtypes_dict = create_all_model_json_dict(local=True,
    path="/home/heiko/Repos/SKlearn_to_MLFLow/model_dump",
    features="feature_dtypes.json")
feature_dtypes_dict


feature_dtypes_dict = create_all_model_json_dict(local=False,
    list_of_models=["CI_polymer", "MFI_polymer"],
    features="feature_dtypes.json")
# feature_dtypes_dict = None
feature_dtypes_dict


dtype_dict=flatten_dict(nested_dict=feature_dtypes_dict)
dtype_dict



# Data for Model: 
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
data.dtypes

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

data = decode_df_mlflow_dtype(data = data, dtype_dict=dtype_dict)
data2 = decode_df_mlflow_dtype(data = data2, dtype_dict=dtype_dict)
data3 = decode_df_mlflow_dtype(data = data3, dtype_dict=dtype_dict)

data
dtype_dict
data.dtypes

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




# azure mlflow.pyfunc.load_model()
# model= get_mlflow_model(model_name="MFI_polymer", azure=True, model_dir = None)



# Data for MFI & CI polymer model
# an efficient way to use MLflow to make predicitons and 


M_per = 2  # 0-3.67
Xf = 15  # 13.45 - 18.4
SA = 61.2 #52 - 79.7


def create_polymer_data(M_per, Xf, SA):
    SASA = SA**2
    SASASA = SA**3
    XfXf = Xf**2
    XfXfXf = Xf**3
    return pd.DataFrame(
        data=[[M_per, Xf, SA, SASA, SASASA, XfXf, XfXfXf]],
            columns=["M%", "Xf", "SA", "SASA", "SASASA", "XfXf", "XfXfXf"],
        )


polymer_data = create_polymer_data(M_per = 2, Xf = 15, SA = 61.2)
polymer_data


load_dotenv()
local_run = os.getenv("LOCAL_RUN", False)

get_model_json_artifact(model_name= "MFI_polymer", features="feature_dtypes.json")
get_model_json_artifact(model_name= "MFI_polymer", features="feature_limits.json")

polymer_data_decoded = decode_df_mlflow_dtype(data = polymer_data, dtype_dict=get_model_json_artifact(model_name= "MFI_polymer", features="feature_dtypes.json"))
polymer_data_decoded.dtypes

MFI_model= get_mlflow_model(azure=True, model_name="MFI_polymer")

MFI_model.predict(polymer_data_decoded)


create_warning(TAG_limit_dict=get_model_json_artifact(model_name= "MFI_polymer", features="feature_limits.json"), key = "M%", value=4) #yes
create_warning(TAG_limit_dict=get_model_json_artifact(model_name= "MFI_polymer", features="feature_limits.json"), key = "M%", value=2) #no

