# Databricks notebook source
# MAGIC %md
# MAGIC Machine Learning End to End Process
# MAGIC <img title="ML E2E" alt="Machine Learning End to End Process" src="https://nhathoang-public-bucket.s3.ap-southeast-1.amazonaws.com/images/mlops-uc-end2end-0.jpg">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import libraries

# COMMAND ----------

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature


# COMMAND ----------

# MAGIC %md
# MAGIC ### Define preprocessing function 

# COMMAND ----------

def preprocess_data(df):
    cat_cols = ['hour', 'am_or_pm', 'day_of_week']

    num_cols = [
        'pickup_longitude', 'pickup_latitude', 
        'dropoff_longitude', 'dropoff_latitude',
        'passenger_count', 'distance'
    ]

    def calculate_distance(df, lat1, lon1, lat2, lon2):
        r = 6371
        phi1 = np.radians(df[lat1])
        phi2 = np.radians(df[lat2])

        delta_phi = np.radians(df[lat1] - df[lat2])
        delta_lambda = np.radians(df[lon1] - df[lon2])

        a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = (r * c)

        return d

    process = df.copy()
    process['distance'] = calculate_distance(
        process, 
        'pickup_latitude', 'pickup_longitude', 
        'dropoff_latitude', 'dropoff_longitude'
    )
    process['pickup_datetime'] = process['pickup_datetime'] - pd.Timedelta(hours=4)
    process['day_of_week'] = process['pickup_datetime'].dt.strftime('%a')
    process['hour'] = process['pickup_datetime'].dt.hour
    process['am_or_pm'] = np.where(process['hour'] < 12, 'am', 'pm')

    for col in cat_cols:
        process[col] = process[col].astype('category')

    caterogies = np.stack(
        [process[col].cat.codes.values for col in cat_cols], axis=1
    )

    caterogies = torch.tensor(caterogies, dtype=torch.int64)
    category_sizes = [len(process[col].cat.categories) for col in cat_cols]
    
    numerics = torch.tensor(process[num_cols].values, dtype=torch.float)

    process = torch.cat([caterogies, numerics], 1)

    return process, category_sizes

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read data

# COMMAND ----------

df = spark.read.csv(
    '/Volumes/delta_lake/bronze/taxi_fare/*',
    inferSchema=True, header=True
).toPandas()

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Split to train / test set

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['fare_amount'], axis=1), 
    df['fare_amount'], 
    test_size=0.2
)

y_train = torch.tensor(y_train.values, dtype=torch.float).reshape(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float).reshape(-1, 1)

X_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Convert to Pytorch Tensor

# COMMAND ----------

X_train, category_sizes = preprocess_data(X_train)

X_test, _ = preprocess_data(X_test)

embedding_sizes = [(size, min(50, (size + 1) // 2)) for size in category_sizes]

print(category_sizes)
print(embedding_sizes)
X_train[:5]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create model class

# COMMAND ----------

class TabularModel(nn.Module):
    def __init__(self, embedding_sizes, numeric_size, output_size, layers, p=0.5):
        super().__init__()
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_categories_input, embedding_size) for num_categories_input, embedding_size in embedding_sizes
        ])
        self.embedding_drop = nn.Dropout(p)
        self.normal_numeric = nn.BatchNorm1d(numeric_size)

        layer_list = []
        categorical_size = sum([embedding_size for _, embedding_size in embedding_sizes])
        input_size = categorical_size + numeric_size

        for i in layers:
            layer_list.append(nn.Linear(input_size, i))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.BatchNorm1d(i))
            layer_list.append(nn.Dropout(p))
            input_size = i
        
        layer_list.append(nn.Linear(layers[-1], output_size))
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        x_categories = x[:, :3].type(torch.LongTensor)
        x_numeric = x[:, 3:]
        x_embeddings = []
        for i, embedding in enumerate(self.embedding_layers):
            x_embeddings.append(embedding(x_categories[:, i]))
        x = torch.cat(x_embeddings, 1)
        x = self.embedding_drop(x)

        x_numeric = self.normal_numeric(x_numeric)

        x = torch.cat([x, x_numeric], 1)
        x = self.layers(x)

        return x

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create model instance, loss & optimize functions

# COMMAND ----------

model = TabularModel(embedding_sizes, 6, 1, [200, 100], 0.5)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# COMMAND ----------

model

# COMMAND ----------

X_train[:5]

# COMMAND ----------

y_train

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train model

# COMMAND ----------

epochs = 300
losses = []

for i in range(epochs):
    y_pred = model(X_train)
    loss = torch.sqrt(criterion(y_pred, y_train))
    losses.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i == 0) or ((i + 1)%10 == 0):
        print(f'Epoch {i + 1}: {loss}')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluate model with train / test set

# COMMAND ----------


with torch.no_grad():
  y_train_pred = model(X_train)
  y_test_pred = model(X_test)


rmse_train = mean_squared_error(y_train.numpy(), y_train_pred.detach().numpy())**0.5
r2_train = r2_score(y_train.numpy(), model(X_train).detach().numpy())

rmse_test = mean_squared_error(y_test.numpy(), y_test_pred.detach().numpy())**0.5
r2_test = r2_score(y_test.numpy(), model(X_test).detach().numpy())

print('For train set')
print(f'RMSE: {rmse_train}')
print(f'R2: {r2_train}')
print('-------------------')
print('For test set')
print(f'RMSE: {rmse_test}')
print(f'R2: {r2_test}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logging model

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create signature

# COMMAND ----------

signature = infer_signature(X_train.numpy(), model(X_train).detach().numpy())
signature

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create experiment

# COMMAND ----------

# experiment_name = '/Users/admin@lhngift.com/Taxi_Fare_Prediction'
# experiment_id = mlflow.create_experiment(experiment_name)

# COMMAND ----------

xp_path = '/Users/admin@lhngift.com/Taxi_Fare_Prediction'

experiment_id = mlflow.search_experiments(
    filter_string=f"name LIKE '{xp_path}%'", order_by=["last_update_time DESC"]
)[0].experiment_id

display(
    mlflow.search_experiments(filter_string=f"name LIKE '{xp_path}%'", order_by=["last_update_time DESC"])
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Log model

# COMMAND ----------

with mlflow.start_run(experiment_id=experiment_id) as run:
    mlflow.log_param('n_epochs', epochs)
    mlflow.log_param('lr', 0.001)
    mlflow.log_metric('RMSE train', rmse_train)
    mlflow.log_metric('R2 train', r2_train)
    mlflow.log_metric('RMSE test', rmse_test)
    mlflow.log_metric('R2 test', r2_test)
    mlflow.pytorch.log_model(model, 'model', signature=signature)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load model & make prediction with PyTorch favour

# COMMAND ----------

logged_model = 'runs:/fc96f3f9bc584326a36f353f075d2922/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pytorch.load_model(logged_model)

loaded_model(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load model & make prediction with PyFunc favour

# COMMAND ----------

# Predict on a Pandas DataFrame.
loaded_model = mlflow.pyfunc.load_model(logged_model)

loaded_model.predict(pd.DataFrame(X_test))


# COMMAND ----------


