

import os
from dotenv import load_dotenv

import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import mlflow.pyfunc as ml_py_load
import mlflow.sklearn as ml_sk_load
import pickle


from azure.storage.blob import BlobServiceClient
import json



load_dotenv()




client = MlflowClient()
for rm in client.list_registered_models():
    pprint(dict(rm), indent=4)



model_name = "dashapp_model"
stage = "Staging"


#pyfunc
model = mlflow.pyfunc.load_model(model_uri= f"models:/{model_name}/{stage}")
model.metadata.run_id

# sklearn
model = mlflow.sklearn.load_model(model_uri= f"models:/{model_name}/{stage}")
model.metadata.run_id




from dotenv import load_dotenv

load_dotenv()

import mlflow.pyfunc


model_name = "dashapp_model"
stage = "Staging"

#pyfunc
model = mlflow.pyfunc.load_model(model_uri= f"models:/{model_name}/{stage}")
model_id = model.metadata.run_id
model_id


connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
container_name = os.getenv("CONTAINER_NAME", "model-container")

connection_string
container_name

def find_and_get_json_from_blob(connection_string, container_name, model_id, file_name):
    block_blob_service= BlobServiceClient.from_connection_string(connection_string)
    container_client = block_blob_service.get_container_client(container_name)

    for blob in container_client.list_blobs():
        if model_id in blob.name and file_name in blob.name:
            output = json.loads(container_client.download_blob(blob).readall().decode("utf-8"))
            return output


feature_limits = find_and_get_json_from_blob(connection_string, container_name, model_id, file_name = "feature")
feature_limits 





connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
container_name = os.getenv("CONTAINER_NAME", "model-container")

connection_string
container_name


model_id = model.metadata.run_id
model_id
model_name = "dashapp_model"
local_path = "/home/heiko/Repos/SKlearn_to_MLFLow/model_dump"


def mlflow_model_to_local_dumper(connection_string, container_name, model_name, model_id, local_path):
    block_blob_service= BlobServiceClient.from_connection_string(connection_string)
    container_client = block_blob_service.get_container_client(container_name)

    for blob in container_client.list_blobs():
        if model_id in blob.name and "artifact" in blob.name: 
            download_path = os.path.join(local_path, model_name, blob.name.split("/")[-1])
            bytes = container_client.get_blob_client(blob).download_blob().readall()

            os.makedirs(os.path.dirname(download_path), exist_ok = True)

            with open(download_path, "wb") as file:
                file.write(bytes)




# https://pretagteam.com/question/download-all-blobs-files-locally-from-azure-container-using-python
# download_blobs.py
# Python program to bulk download blob files from azure storage
# Uses latest python SDK() for Azure blob storage
# Requires python 3.6 or above
import os
from azure.storage.blob
import BlobServiceClient, BlobClient
from azure.storage.blob
import ContentSettings, ContainerClient


class AzureBlobFileDownloader:
    def __init__(self):
        print("Intializing AzureBlobFileDownloader")

        # Initialize the connection to Azure storage account
        self.blob_service_client = BlobServiceClient.from_connection_string(MY_CONNECTION_STRING)
        self.my_container = self.blob_service_client.get_container_client(MY_BLOB_CONTAINER)

    def save_blob(self, file_name, file_content):
        # Get full path to the file
        download_file_path = os.path.join(LOCAL_BLOB_PATH, file_name)

        # for nested blobs, create local path as well!
        os.makedirs(os.path.dirname(download_file_path), exist_ok = True)

        with open(download_file_path, "wb") as file:
            file.write(file_content)

    def download_all_blobs_in_container(self):
        my_blobs = self.my_container.list_blobs()
        for blob in my_blobs:
            print(blob.name)
            bytes = self.my_container.get_blob_client(blob).download_blob().readall()
            self.save_blob(blob.name, bytes)



MY_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")

# Replace with blob container
MY_BLOB_CONTAINER = model_id #os.getenv("CONTAINER_NAME", "model-container")
model_id

# Replace with the local folder where you want files to be downloaded
LOCAL_BLOB_PATH = "/home/heiko/Repos/SKlearn_to_MLFLow/model_dump"


# Initialize class and upload files
azure_blob_file_downloader = AzureBlobFileDownloader()
azure_blob_file_downloader.download_all_blobs_in_container()




################
################

import os
from functools import lru_cache
from pathlib import PurePosixPath
from pathlib import PurePath
from mlflow.pyfunc import load_model as ml_py_load
# from mlflow.sklearn import load_model as ml_sk_load

from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=None)
def get_mlflow_model(model_name):
    # model_name = "dashapp_model"
    model_dir = os.getenv("MLFLOW_MODEL_DIRECTORY", "models:")
    model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Staging")

    #f"models:/{model_name}/{stage}"

    artifact_path = PurePosixPath(model_dir).joinpath(model_name, model_stage)
    print(artifact_path)

    model = ml_py_load(str(artifact_path))

    return model



model = get_mlflow_model(model_name = "dashapp_model")

model.metadata.run_id


