import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import os

folder = "A2_Experiment data"

files = [
"Swift_Creek_Inflow_Data_FILLED.csv",
"Bunchgrass_Meadow_Inflow_Data_FILLED.csv",
"Touchet_Inflow_Data_FILLED.csv",
"Paradise_Inflow_Data_FILLED.csv",
"Easy_Pass_Inflow_Data_FILLED.csv"
]

for file in files:

    df = pd.read_csv(folder + "/" + file)

    # persistence prediction
    prediction = df["DISCHRG_LAG"]
    true = df["DISCHRG"]

    rmse = np.sqrt(((true - prediction) ** 2).mean())
    mae = np.abs(true - prediction).mean()
    mre = (np.abs((true - prediction) / true)).mean() * 100
    r2 = r2_score(true, prediction)

    print(file)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("MRE:", mre)
    print("R2:", r2)
    print()