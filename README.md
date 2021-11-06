# SKlearn_to_MLFLow
In this Repo a simple Sklearn Model will be trained and pushed to MLFlow


## Install

This Repo is based on poetry

```bash

python3 -m venv .venv

# switch manually to virtual environment and then

$(.venv) poetry install 
# will install all dependencies from the pyproject.toml file

```

## General_training

The general Traning pipeline script is configured by the `.yaml` file

The yaml file needs to be modified, since the path to the data is hardcoded  there.


## Data Preprocessing

The data gets prefiltered by the `filter.py` configured by `filter_config.yaml`  in the folder filter and gets stored in the `data` folder from where it gets loaded.


## ENV File

I have MLFLow in a Docker Container running with a Azurite and Postgres instance. Therefor I need a connection string to connect to MLFlow containers. This is in the `.venv` file as well as the connection to MLFlow.

I will link later the Repo for the MLFlow Docker Container.


