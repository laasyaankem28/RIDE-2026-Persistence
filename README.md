# RIDE-2026-Persistence

# Report #2 – Persistence Model Training

## 1. Introduction

In this experiment, a **persistence model** was used to predict streamflow discharge using the **A_2 experiment dataset**. The persistence model is commonly used as a baseline in hydrological forecasting because it assumes that the future discharge value will be similar to the most recent observed value.

The model was evaluated using several performance metrics:

- **MSE (Mean Squared Error)** – measures the average squared difference between predicted and actual values.
- **RMSE (Root Mean Squared Error)** – the square root of MSE, representing prediction error in the same units as discharge.
- **MRE (Mean Relative Error)** – measures the relative magnitude of prediction errors.
- **R² (Coefficient of Determination)** – indicates how well the model explains the variance in observed discharge values.

### Input Features

The dataset includes the following input variables:

- **WESD** – Water Equivalent Snow Depth
- **GHI** – Global Horizontal Irradiance
- **PRCP** – Precipitation
- **SNWD** – Snow Depth
- **TAVG** – Average Temperature

The **target variable** for prediction is **discharge**.

---

## 2. Methodology

The persistence model predicts the discharge at time `t` using the discharge value from the previous time step `t-1`.

Q*t = Q*(t-1)

Where:

- `Q_t` = predicted discharge at time `t`
- `Q_(t-1)` = observed discharge at the previous time step

Although simple, persistence models are useful as **baseline models** to compare against more advanced machine learning methods.

Each dataset was processed separately, and prediction performance was evaluated using the metrics described earlier.

---

## 3. Results

| Dataset                                  | RMSE   | MAE    | MRE    | R²    |
| ---------------------------------------- | ------ | ------ | ------ | ----- |
| Swift_Creek_Inflow_Data_FILLED.csv       | 4.764  | 0.768  | 8.497  | 0.513 |
| Bunchgrass_Meadow_Inflow_Data_FILLED.csv | 70.886 | 48.661 | 8.462  | 0.981 |
| Touchet_Inflow_Data_FILLED.csv           | 19.134 | 3.495  | inf    | 0.665 |
| Paradise_Inflow_Data_FILLED.csv          | 4.376  | 1.351  | 13.055 | 0.653 |
| Easy_Pass_Inflow_Data_FILLED.csv         | 17.332 | 6.168  | 19.925 | 0.431 |

---

## 4. Observations

The results show that the performance of the persistence model varies across datasets.

- **Bunchgrass Meadow** achieved the highest **R² value (0.981)**, indicating that the persistence model explains most of the variance in discharge values at this location. This suggests that discharge values change gradually over time.

- **Touchet dataset** produced an **infinite MRE value**. This likely occurred because some actual discharge values were **zero**, which results in division by zero when computing relative error.

- **Easy Pass** had the lowest **R² value (0.431)**, indicating that the persistence model is less effective for this dataset due to more dynamic variations in streamflow.

Overall, the persistence model performs reasonably well for datasets where discharge values change slowly but performs worse for datasets with rapid fluctuations.

---

## 5. Conclusion

The persistence model serves as a simple baseline method for streamflow prediction. While it performs well in locations where discharge changes gradually over time, it is less effective for highly variable datasets.

Future models could improve prediction accuracy by incorporating machine learning techniques that better utilize the available input features such as precipitation, snow depth, and temperature.
