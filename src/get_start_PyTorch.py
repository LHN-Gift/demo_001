# Databricks notebook source
# MAGIC %md
# MAGIC PyTorch Workflow
# MAGIC <img title="PyTorch" alt="PyTorch Workflow" src="https://nhathoang-public-bucket.s3.ap-southeast-1.amazonaws.com/images/pytorch+workflow.jpg">

# COMMAND ----------

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets

import plotly.graph_objects as go



# COMMAND ----------

X = torch.linspace(1, 50, 50).reshape(50, 1)
print(X.shape)
X[:5]

# COMMAND ----------

error = torch.randint(-9, 9, (50, 1), dtype=torch.float)
error[:5]

# COMMAND ----------

y = 2 * X + 5 + error
y[:5]

# COMMAND ----------

fig = go.Figure()

data = pd.DataFrame(
    data=np.concatenate([X.numpy(), y.numpy(), (2 * X + 5).numpy()], axis=1),
    columns=['X', 'y', 'y_theory']
)

fig.add_trace(
    go.Scatter(x=data['X'], y=data['y'], mode='markers', name='practical')
)

fig.add_trace(
    go.Scatter(x=data['X'], y=data['y_theory'], mode='lines', name='theory')
)

fig.update_layout(
    title=dict(text='Data visualization', x=0.5),
    height=600,
    width=900
)

fig.show()

# COMMAND ----------

torch.manual_seed(42)

model = nn.Linear(in_features=1, out_features=1)

print(model.weight)
print(model.bias)

# COMMAND ----------

class Model( nn.Module ):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
    

# COMMAND ----------

torch.manual_seed(42)

model = Model(in_features=1, out_features=1)

print(model.linear.weight)
print(model.linear.bias)

# COMMAND ----------

fig = go.Figure()

data = pd.DataFrame(
    data=np.concatenate(
        [
            X.numpy(), 
            y.numpy(), 
            (2 * X + 5).numpy(),
            model.forward(X).detach().numpy()
        ], 
        axis=1
    ),
    columns=['X', 'y', 'y_theory', 'y_pred']
)

fig.add_trace(
    go.Scatter(x=data['X'], y=data['y'], mode='markers', name='practical')
)

fig.add_trace(
    go.Scatter(x=data['X'], y=data['y_theory'], mode='lines', name='theory')
)

fig.add_trace(
    go.Scatter(x=data['X'], y=data['y_pred'], mode='lines', name='predict')
)

fig.update_layout(
    title=dict(text='Data visualization', x=0.5),
    height=600,
    width=900
)

fig.show()

# COMMAND ----------

criterion = nn.MSELoss()

# COMMAND ----------

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# COMMAND ----------

epochs = 50
losses = []
weights = []
biass = []

for i in range(epochs):
    y_pred = model.forward(X)
    loss = criterion(y_pred, y)
    losses.append(loss)

    print(f'Epoch {i+1}/{epochs} | Loss: {loss.item():.4f}')
    for name, param in model.named_parameters():
        print(f' | {name}: {param.item():.4f}')
        if name == 'linear.weight':
            weights.append(param.item())
        else:
            biass.append(param.item())
    print('------------------------------------')

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

# COMMAND ----------

fig = go.Figure()

data = pd.DataFrame(
    data=np.concatenate(
        [
            X.numpy(), 
            y.numpy(), 
            (2 * X + 5).numpy()
        ], 
        axis=1
    ),
    columns=['X', 'y', 'y_theory']
)

fig.add_trace(
    go.Scatter(x=data['X'], y=data['y'], mode='markers', name='practical')
)

fig.add_trace(
    go.Scatter(x=data['X'], y=data['y_theory'], mode='lines', name='theory', line=dict(color='blue'))
)

colors = ['red', 'purple',  'orange', 'yellow', 'green']
j = 0
for i in [0, 1, 2, 3, 49]:
    data[f'y_pred_{i}'] = data['X'] * weights[i] + biass[i]
    fig.add_trace(
        go.Scatter(
            x=data['X'], y=data[f'y_pred_{i}'], 
            mode='lines', name=f'predict_{i}', line=dict(color=colors[j])
        )
    )
    j += 1

fig.update_layout(
    title=dict(text='Data visualization', x=0.5),
    height=900,
    width=1200
)

fig.show()


# COMMAND ----------

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=list(range(epochs)), 
        y=[i.item() for i in losses], 
        mode='lines')
)



fig.update_layout(
    title=dict(text='Loss vs. Epoch', x=0.5),
    height=600,
    width=600
)

fig.show()


# COMMAND ----------

losses[0].item()

# COMMAND ----------

X_test = torch.tensor([4, 5], dtype=torch.float).reshape(2, 1)

# COMMAND ----------

model(X_test)

# COMMAND ----------


