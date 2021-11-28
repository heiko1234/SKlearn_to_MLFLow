




There is a `.env` file that contains:

```bash

MLFLOW_TRACKING_URI="172.19.0.4:5000" 
#MLFLOW_TRACKING_URI="mlflow_general_nw:5000" 
#Ja, ein bissel, läuft durch, aber nicht sichtbar

#MLFLOW_TRACKING_URI="http://localhost:5000"

```


Type into the console:

```bash

docker build -t mlflow_training .

```



To check your mlflow_network:

```bash

docker network ls

docker network inspect mlflow_general_nw

```
result in 

```bash

"Name": "mlflow_mlflow_1",
"EndpointID": "8841b67f20d1631590d5526cb3ecdc5d1e42bfebfdcc9402db7f44aea342d739",
"MacAddress": "02:42:ac:13:00:04",
"IPv4Address": "172.19.0.4/16",
"IPv6Address": ""

```


and now lets run the container and bring the artifacts to mlflow in our local docker container.

```bash

# in the .env file
# MLFLOW_TRACKING_URI="mlflow_general_nw:5000" 

docker run --net=bridge mlflow_training

```

or

```bash

ip addr show docker0

```


** ATTENTION**

I m sorry! Locally the models trained in the docker don't show up.
Training goes with no errors but the don t show up in the artifact storage or in the local instance of mlflow.

If you execute the general_training.py file alone it graps all information of .env and they show up.



