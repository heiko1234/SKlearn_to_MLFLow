
# This file will download all files from specified models that are Staged



import os
import yaml
from pathlib import PurePosixPath
from pathlib import Path
from mlflow.pyfunc import load_model as ml_py_load

# from mlflow.sklearn import load_model as ml_sk_load

from azure.storage.blob import BlobServiceClient

from dotenv import load_dotenv

load_dotenv()


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


def get_mlflow_model(model_name):
    model_dir = os.getenv("MLFLOW_MODEL_DIRECTORY", "models:")
    model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Staging")

    artifact_path = PurePosixPath(model_dir).joinpath(model_name, model_stage)

    model = ml_py_load(str(artifact_path))

    return model


def all_run_ids(model_names):
    output = {}
    for model in model_names:
        try:
            model_id = get_mlflow_model(model_name="dashapp_model").metadata.run_id
            output[model] = model_id
        except:
            continue
    return output


def mlflow_model_to_local_dumper(
    connection_string, container_name, model_name, model_id, local_path
):
    block_blob_service = BlobServiceClient.from_connection_string(
        connection_string
    )
    container_client = block_blob_service.get_container_client(container_name)

    for blob in container_client.list_blobs():
        if model_id in blob.name and "artifact" in blob.name:
            download_path = os.path.join(
                local_path, model_name, blob.name.split("/")[-1]
            )
            bytes = (
                container_client.get_blob_client(blob)
                .download_blob()
                .readall()
            )

            os.makedirs(os.path.dirname(download_path), exist_ok=True)

            with open(download_path, "wb") as file:
                file.write(bytes)



def main():

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    container_name = os.getenv("CONTAINER_NAME", "model-container")

    path = Path(__file__).parent
    configuration = read_configuration(
        configuration_file_path=os.path.join(
            path, "dump_mlflow_model_config.yaml"
        )
    )
    model_list = configuration["model_list"]
    local_path = configuration["local_path"]

    run_id_dicts = all_run_ids(model_names=model_list)
    for model in model_list:
        try:
            print(f"Started to dump: {model} into {local_path}")
            mlflow_model_to_local_dumper(
                connection_string=connection_string,
                container_name=container_name,
                model_name=model,
                model_id=run_id_dicts[model],
                local_path=local_path,
            )
        except:
            continue
    print("Models are downloaded from mlflow")


if __name__ == "__main__":

    main()
