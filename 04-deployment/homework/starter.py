#!/usr/bin/env python
# coding: utf-8

# In[16]:


# get_ipython().system('pip freeze | grep scikit-learn')


# In[17]:


# get_ipython().system('python -V')


# In[33]:

import sys
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import os


# In[19]:
year = int(sys.argv[1])
month = int(sys.argv[2])
taxi_type = 'green'

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

RUN_ID = os.getenv('RUN_ID', 'e1efc53e9bd149078b0c12aeaa6365df')


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[20]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[21]:


df = read_data(input_file)


# In[28]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_val = df.duration.values
y_pred = model.predict(X_val)
mean_duration = np.mean(y_pred)
print("mean_duration", mean_duration)
std_dev = np.std(y_pred)
print(std_dev)


# In[44]:


def apply_model(input_file, run_id, output_file):
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()
    # df_result['ride_id'] = df['ride_id']
    # # df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    # df_result['PULocationID'] = df['PULocationID']
    # df_result['DOLocationID'] = df['DOLocationID']
    # df_result['actual_duration'] = df['duration']
    # df_result['predicted_duration'] = y_pred
    # df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    # df_result['model_version'] = run_id
    
    df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[45]:

# In[46]:


apply_model(input_file=input_file, run_id=RUN_ID, output_file=output_file)

