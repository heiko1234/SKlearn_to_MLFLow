ARG PYTHON_VERSION="3.8"
FROM python:${PYTHON_VERSION} as foundation

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=$PIP_INDEX_URL



RUN pip3 install --upgrade pip
RUN mkdir -p /usr/app/
RUN mkdir -p /usr/app/.git/
RUN mkdir -p /usr/app/script/
RUN mkdir -p /usr/data/


COPY /docker_train_model/script/ /usr/app/script/
COPY /docker_train_model/.env /usr/app/
COPY /data/ChemicalManufacturingProcess.parquet /usr/data/
COPY /pyproject.toml /usr/app/
COPY /poetry.lock /usr/app/
COPY /docker_train_model/poetry.toml /usr/app/
COPY /general_training.py /usr/app/script/
COPY /training_config.yaml /usr/app/script/

WORKDIR /usr/app/

RUN pip install poetry toml \
    && poetry install --no-dev

## slim image
FROM python:${PYTHON_VERSION}-slim

COPY --from=foundation /usr/app /usr/app
COPY --from=foundation /usr/data /usr/data

ENV PATH=/usr/app/.venv/bin:$PATH
ENV PYTHONUNBUFFERED 1

# 
#ARG AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://localhost:10000/devstoreaccount1;QueueEndpoint=http://localhost:10001/devstoreaccount1"
#ARG MLFLOW_TRACKING_URI="http://localhost:5000"


RUN apt-get update \
  && apt-get install -y \
    git \
  && rm -rf /var/lib/apt/lists/*

CMD bash -c "/usr/app/.venv/bin/python /usr/app/script/general_training.py"

