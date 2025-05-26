import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn  # Required for autologging

# import warnings
# warnings.filterwarnings("default")

# import logging
# logging.basicConfig(level=logging.DEBUG)

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('NYC-taxi-experiment')

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        print('fle found', filename)
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    mlflow.sklearn.autolog()  

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run(): 
        print('strated tracking')
        rf = RandomForestRegressor(max_depth=10, random_state=0,n_jobs=1,verbose=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        print('model fitted')

        rmse = mean_squared_error(y_val, y_pred,squared= False)
        print(f"RMSE: {rmse:.3f}")

if __name__ == '__main__':
    run_train()