# save_data

import pandas as pd


data = pd.read_csv("/home/heiko/Repos/SKlearn_to_MLFLow/data/ChemicalManufacturingProcess.csv", sep=";")

data

data.to_parquet("/home/heiko/Repos/SKlearn_to_MLFLow/data/ChemicalManufacturingProcess.parquet")


