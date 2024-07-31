# Databricks notebook source
# MAGIC %md
# MAGIC ### Import libraries

# COMMAND ----------

from datetime import datetime as dt

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

import plotly.graph_objects as go



# COMMAND ----------

# MAGIC %md
# MAGIC ### Read and get basic understanding of data

# COMMAND ----------

df = spark.read.csv(
    '/Volumes/delta_lake/bronze/taxi_fare/*',
    inferSchema=True, header=True
).toPandas()

df.head()

# COMMAND ----------

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualization of target column

# COMMAND ----------

fig = go.Figure()

fig.add_trace(
    go.Histogram(
        x=df['fare_amount'], nbinsx=30,
        marker={'line': {'width': 1, 'color': 'white'}}
    )
)

fig.update_layout(
    title=dict(text='Histogram of Fare Amount', x=0.5),
    xaxis_title='Fare Amount', yaxis_title='Count',
    width=800, height=600
)

fig.show()

# COMMAND ----------

fig = go.Figure()

fig.add_trace(
    go.Histogram(
        x=np.log(df['fare_amount']), nbinsx=30,
        marker={'line': {'width': 1, 'color': 'white'}}
    )
)

fig.update_layout(
    title=dict(text='Histogram of Logarit (Fare Amount)', x=0.5),
    xaxis_title='Fare Amount', yaxis_title='Count',
    width=800, height=600
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualization data by location

# COMMAND ----------

import plotly.graph_objects as go

mapbox_access_token = 'pk.eyJ1IjoibmhhdGhvYW5nMjkxMCIsImEiOiJjbHdmenJ3dGkyMXl0Mmptbm9kYWc2eW1xIn0.qvPWqIGAcw94rhB0sNS8Uw'

fig = go.Figure(go.Scattermapbox(
        lat=df['pickup_latitude'],
        lon=df['pickup_longitude'],
        mode='markers',
        opacity=0.5,
        marker=go.scattermapbox.Marker(
            size=10
        ),
    ))

fig.update_layout(
    title=dict(text='NYC Taxi Pickup location', x=0.5),
    height=800, width=1200,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=df['pickup_latitude'].mean(),
            lon=df['pickup_longitude'].mean()
        ),
        pitch=0,
        zoom=9
    )
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create function to calculate distance between pickup and dropoff

# COMMAND ----------

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

# COMMAND ----------

df['distance'] = calculate_distance(
    df, 
    'pickup_latitude', 'pickup_longitude', 
    'dropoff_latitude', 'dropoff_longitude'
)

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualization Distance vs. Fare Amount

# COMMAND ----------

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df['distance'], y=df['fare_amount'],
        mode='markers'
    )
)

fig.update_layout(
    title=dict(text='Distance Vs. Fare Amount', x=0.5),
    width=800, height=600,    
    xaxis_title='Distance (Km)', yaxis_title='Fare Amount'
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transfroming data

# COMMAND ----------

# MAGIC %md
# MAGIC #### Convert UTC time to New York time

# COMMAND ----------

df['pickup_datetime'] = df['pickup_datetime'] - pd.Timedelta(hours=4)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create dat_of_week, hour and am_or_pm columns

# COMMAND ----------

df['day_of_week'] = df['pickup_datetime'].dt.strftime('%a')
df['hour'] = df['pickup_datetime'].dt.hour
df['am_or_pm'] = np.where(df['hour'] < 12, 'am', 'pm')

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Convert to category type

# COMMAND ----------

cat_cols = ['hour', 'am_or_pm', 'day_of_week']

num_cols = [
    'pickup_longitude', 'pickup_latitude', 
    'dropoff_longitude', 'dropoff_latitude',
    'passenger_count', 'distance'
]

y_col = ['fare_amount']


# COMMAND ----------

for col in cat_cols:
    df[col] = df[col].astype('category')

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preparing data for PyTorch

# COMMAND ----------

# MAGIC %md
# MAGIC #### Categrories data

# COMMAND ----------

categories = np.stack(
    [df[col].cat.codes.values for col in cat_cols], axis=1
)

categories = torch.tensor(categories, dtype=torch.int64)

categories

# COMMAND ----------

# MAGIC %md
# MAGIC #### Numeric data

# COMMAND ----------

numerics = torch.tensor(df[num_cols].values, dtype=torch.float)

numerics

# COMMAND ----------

# MAGIC %md
# MAGIC #### Target data

# COMMAND ----------

label = torch.tensor(df[y_col].values, dtype=torch.float)

label

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define category sizes and embedding sizes

# COMMAND ----------

category_sizes = [len(df[col].cat.categories) for col in cat_cols]

embedding_sizes = [(size, min(50, (size + 1) // 2)) for size in category_sizes]

print(cat_cols)
print(category_sizes)
print(embedding_sizes)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preparing simulation

# COMMAND ----------

# MAGIC %md
# MAGIC #### Category columns

# COMMAND ----------

cat_test = caterogies[: 5]
cat_test

# COMMAND ----------

embedding_test = nn.ModuleList([
    nn.Embedding(num_cat_in, embedding_dimension) for num_cat_in, embedding_dimension in embedding_sizes
])

embedding_test

# COMMAND ----------

embedding_result = []

for i, embedding in enumerate(embedding_test):
    embedding_result.append(embedding(cat_test[:, i]))

embedding_result

# COMMAND ----------

embedding_result = torch.cat(embedding_result, 1)
print(embedding_result.shape)
embedding_result

# COMMAND ----------

# MAGIC %md
# MAGIC #### Numeric columns

# COMMAND ----------

num_test = numerics[:5]
num_test

# COMMAND ----------

normal_test = nn.BatchNorm1d(6)
normal_test

# COMMAND ----------

normal_result = normal_test(num_test)
print(normal_result.shape)
normal_result

# COMMAND ----------

# MAGIC %md
# MAGIC #### Concat all data of category and numeric

# COMMAND ----------

test_result = torch.cat([embedding_result, normal_result], 1)
print(test_result.shape)
test_result

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create class for model

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

    def forward(self, x_categories, x_numeric):
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
# MAGIC #### Create instance of model

# COMMAND ----------

torch.manual_seed(42)

model = TabularModel(embedding_sizes, len(num_cols), 1, [200, 100], 0.5)

model

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create loss and optimize functions

# COMMAND ----------

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train model

# COMMAND ----------

categories[:5]

# COMMAND ----------

numerics[:5]

# COMMAND ----------

label

# COMMAND ----------

epochs = 300
losses = []

for i in range(epochs):
    y_pred = model(categories, numerics)
    loss = torch.sqrt(criterion(y_pred, label))
    losses.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%10 == 0:
        print(f'Epoch {i}: {loss}')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualization Loss vs. Epoch

# COMMAND ----------

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=list(range(epochs)),
        y=[i.item() for i in losses]
    )
)

fig.update_layout(
    title={
        'text': 'Error vs. Epoch', 'x': 0.5
    },
    xaxis_title='Epoch',
    yaxis_title='Root Mean Square Error',
    width=900,
    height=600
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model evaluation

# COMMAND ----------

from sklearn.metrics import mean_squared_error, r2_score

# COMMAND ----------

r2 = r2_score(label.numpy(), y_pred.detach().numpy())
RMSE = mean_squared_error(label.numpy(), y_pred.detach().numpy())**0.5

print(f'R2: {r2}')
print(f'Root Mean Square Error: {RMSE}')

# COMMAND ----------

pd.DataFrame(
  data=np.concatenate(
    [label[:10].numpy(), y_pred[:10].detach().numpy()],
    axis=1
  ),
  columns=['label', 'prediction']
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Make some prediction

# COMMAND ----------

with torch.no_grad():
  y_val = model(cat_test, num_test)

y_val

# COMMAND ----------

label[:5]

# COMMAND ----------


