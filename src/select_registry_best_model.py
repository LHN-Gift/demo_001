# Databricks notebook source
import mlflow
from mlflow import MlflowClient

import pandas as pd



# COMMAND ----------

xp_path = '/Users/admin@lhngift.com/Taxi_Fare_Prediction'

experiment_id = mlflow.search_experiments(
    filter_string=f"name LIKE '{xp_path}%'", order_by=["last_update_time DESC"]
)[0].experiment_id

# COMMAND ----------

runs = mlflow.search_runs(experiment_ids=[experiment_id]).sort_values('start_time', ascending=False)
runs

# COMMAND ----------

latest_model_run_id = runs.iloc[0].run_id
print(latest_model_run_id)

# COMMAND ----------

catalog = "delta_lake"
schema = "ml_model"
model_name = f"{catalog}.{schema}.ny_taxi_fare_prediction"

print(f'Registry or push the best model to {model_name}')

# COMMAND ----------

model_detail = mlflow.register_model(f'runs:/{latest_model_run_id}/model', model_name)

# COMMAND ----------

client = MlflowClient()

client.update_registered_model(
  name=model_detail.name,
  description="This model predicts taxi fare in New York City",
)

# COMMAND ----------

client.update_model_version(
    name=model_detail.name,    
    version=model_detail.version,
    description=f"This model used ANN Pytorch with R2 score is {runs['metrics.R2 train'][0]}",
)

# COMMAND ----------

client.set_registered_model_alias(
  name=model_name,
  alias="Challenger",
  version=model_detail.version
)

# COMMAND ----------

model_detail

# COMMAND ----------

try:
    champion = client.get_model_version_by_alias(model_name, "Champion")
    champion_r2_score = mlflow.get_run(champion.run_id).data.metrics["R2 train"]
    challenger_r2_score = mlflow.get_run(model_detail.run_id).data.metrics["R2 train"]
    
    if champion_r2_score > challenger_r2_score:
        print("Champion model is better than Challenger model")
        print("Keep using previous version")
    else:
        print("Firstly, downgrade Champion model to Challenger alias")
        client.set_registered_model_alias(
            name=model_name,
            alias="Challenger",
            version=champion.version
        )
        print("Then, promote Challenger model to Champion alias")
        client.set_registered_model_alias(
            name=model_name,
            alias="Champion",
            version=model_detail.version
        )
except:    
    print("Champion not found")
    print("Set this model to Champion alias")
    # client.set_registered_model_alias(
    #     name=model_name,
    #     alias="Champion",
    #     version=model_detail.version
    # )

# COMMAND ----------


