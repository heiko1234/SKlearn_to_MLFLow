

import os
from dotenv import load_dotenv

import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import mlflow.pyfunc as ml_py_load
import mlflow.sklearn as ml_sk_load


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



def find_and_get_json_from_blob(connection_string, container_name, model_id, file_name):
    block_blob_service= BlobServiceClient.from_connection_string(connection_string)
    container_client = block_blob_service.get_container_client(container_name)

    for blob in container_client.list_blobs():
        if model_id in blob.name and file_name in blob.name:
            output = json.loads(container_client.download_blob(blob).readall().decode("utf-8"))
            return output


feature_limits = find_and_get_json_from_blob(connection_string, container_name, model_id, file_name = "feature")
feature_limits 






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

model


