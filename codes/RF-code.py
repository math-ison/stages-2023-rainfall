#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# # Read in Datasets

# ## Within this workflow, please comment out all other dataframes when running one set of models.

# In[2]:


#Adjust to your local directories
threehour_df = pd.read_csv('3hr_precip_corrected_wlags.csv')
threehour_df = pd.read_csv('3hr_precip_corrected-wenv.csv')
threehour_df = pd.read_csv('3hr_precip_corrected_wenv_wmonth.csv')
threehour_df = pd.read_csv('3hr_precip_corrected_woenv_wlags.csv')


# In[3]:


threehour_df['DateTime'] = pd.to_datetime(threehour_df['DateTime'])


# ## Cycle 1 Data Splits 

# In[4]:


# Specify the start and end datetime values
start_datetime = pd.to_datetime('1/1/2021 0:00')
end_datetime = pd.to_datetime('12/31/2021 21:00')

# Create a new DataFrame with rows within the specified date range
train1_3hr = threehour_df[(threehour_df['DateTime'] >= start_datetime) & (threehour_df['DateTime'] <= end_datetime)].copy()

# Display the resulting DataFrame
train1_3hr


# In[5]:


# Specify the datetime value
target_datetime = pd.to_datetime('12/31/2020 21:00')

# Create a new DataFrame with rows up to the specified datetime
validation1_3hr = threehour_df[threehour_df['DateTime'] <= target_datetime].copy()
validation1_3hr


# In[6]:


# Specify the start and end datetime values
start_datetime = pd.to_datetime('1/1/2022 0:00')
end_datetime = pd.to_datetime('12/27/2022 21:00')

# Create a new DataFrame with rows within the specified date range
test1_3hr = threehour_df[(threehour_df['DateTime'] >= start_datetime) & (threehour_df['DateTime'] <= end_datetime)].copy()

# Display the resulting DataFrame
test1_3hr


# ## Cycle 2 Data Splits

# In[7]:


# Specify the datetime value
target_datetime = pd.to_datetime('5/31/2021 22:00')

# Create a new DataFrame with rows up to the specified datetime
train2_3hr = threehour_df[threehour_df['DateTime'] <= target_datetime].copy()
train2_3hr


# In[ ]:


# Specify the start and end datetime values
start_datetime = pd.to_datetime('6/1/2021 0:00')
end_datetime = pd.to_datetime('12/31/2021 21:00')

# Create a new DataFrame with rows within the specified date range
validation2_3hr = threehour_df[(threehour_df['DateTime'] >= start_datetime) & (threehour_df['DateTime'] <= end_datetime)].copy()

# Display the resulting DataFrame
validation2_3hr


# In[ ]:


# Specify the start and end datetime values
start_datetime = pd.to_datetime('1/1/2022 0:00')
end_datetime = pd.to_datetime('12/27/2022 21:00')

# Create a new DataFrame with rows within the specified date range
test2_3hr = threehour_df[(threehour_df['DateTime'] >= start_datetime) & (threehour_df['DateTime'] <= end_datetime)].copy()

# Display the resulting DataFrame
test2_3hr


# ## Cycle 3 Data Splits

# In[ ]:


# Specify the start and end datetime values
start_datetime = pd.to_datetime('1/1/2022 0:00')
end_datetime = pd.to_datetime('12/27/2022 21:00')

# Create a new DataFrame with rows within the specified date range
train3_3hr = threehour_df[(threehour_df['DateTime'] >= start_datetime) & (threehour_df['DateTime'] <= end_datetime)].copy()

# Display the resulting DataFrame
train3_3hr


# In[ ]:


# Specify the datetime value
target_datetime = pd.to_datetime('12/31/2020 21:00')

# Create a new DataFrame with rows up to the specified datetime
validation1_3hr = threehour_df[threehour_df['DateTime'] <= target_datetime].copy()
validation1_3hr


# In[ ]:


# Specify the start and end datetime values
start_datetime = pd.to_datetime('1/1/2021 0:00')
end_datetime = pd.to_datetime('12/31/2021 21:00')

# Create a new DataFrame with rows within the specified date range
test3_3hr = threehour_df[(threehour_df['DateTime'] >= start_datetime) & (threehour_df['DateTime'] <= end_datetime)].copy()

# Display the resulting DataFrame
test3_3hr


# ## Cycle 4 Data Split 

# In[ ]:


# Specify the start and end datetime values
start_datetime = pd.to_datetime('1/1/2022 0:00')
end_datetime = pd.to_datetime('12/27/2022 21:00')

# Create a new DataFrame with rows within the specified date range
train4_3hr = threehour_df[(threehour_df['DateTime'] >= start_datetime) & (threehour_df['DateTime'] <= end_datetime)].copy()

# Display the resulting DataFrame
train4_3hr


# In[ ]:


# Specify the start and end datetime values
start_datetime = pd.to_datetime('6/1/2021 0:00')
end_datetime = pd.to_datetime('12/31/2021 21:00')

# Create a new DataFrame with rows within the specified date range
validation4_3hr = threehour_df[(threehour_df['DateTime'] >= start_datetime) & (threehour_df['DateTime'] <= end_datetime)].copy()

# Display the resulting DataFrame
validation4_3hr


# In[ ]:


# Specify the datetime value
target_datetime = pd.to_datetime('5/31/2021 22:00')

# Create a new DataFrame with rows up to the specified datetime
test4_3hr = threehour_df[threehour_df['DateTime'] <= target_datetime].copy()
test4_3hr


# ## Cycle 5 Data Split 

# In[ ]:


# Specify the start and end datetime values
start_datetime = pd.to_datetime('6/1/2021 0:00')
end_datetime = pd.to_datetime('12/27/2022 21:00')

# Create a new DataFrame with rows within the specified date range
train5_3hr = threehour_df[(threehour_df['DateTime'] >= start_datetime) & (threehour_df['DateTime'] <= end_datetime)].copy()

# Display the resulting DataFrame
train5_3hr


# In[ ]:


# Specify the datetime value
target_datetime = pd.to_datetime('5/31/2021 22:00')

# Create a new DataFrame with rows up to the specified datetime
test5_3hr = threehour_df[threehour_df['DateTime'] <= target_datetime].copy()
test5_3hr


# ## Cycle 6 Data Split

# In[ ]:


# Specify the datetime value
target_datetime = pd.to_datetime('12/31/2021 22:00')

# Create a new DataFrame with rows up to the specified datetime
train6_3hr = threehour_df[threehour_df['DateTime'] <= target_datetime].copy()
train6_3hr


# In[ ]:


# Specify the start and end datetime values
start_datetime = pd.to_datetime('1/1/2022 0:00')
end_datetime = pd.to_datetime('12/27/2022 21:00')

# Create a new DataFrame with rows within the specified date range
test6_3hr = threehour_df[(threehour_df['DateTime'] >= start_datetime) & (threehour_df['DateTime'] <= end_datetime)].copy()

# Display the resulting DataFrame
test6_3hr


# ## Cycle 7 Data Split 

# In[25]:


# Specify the datetime value
target_datetime = pd.to_datetime('12/31/2020 22:00')

# Create a new DataFrame with rows up to the specified datetime
train7_3hr_1 = threehour_df[threehour_df['DateTime'] <= target_datetime].copy()

# Specify the start and end datetime values
start_datetime = pd.to_datetime('1/1/2022 0:00')
end_datetime = pd.to_datetime('12/27/2022 21:00')

# Create a new DataFrame with rows within the specified date range
train7_3hr_2 = threehour_df[(threehour_df['DateTime'] >= start_datetime) & (threehour_df['DateTime'] <= end_datetime)].copy()

# Concatenate both dataframes
train7_3hr = pd.concat([train7_3hr_1, train7_3hr_2], ignore_index=True)


# Display the resulting DataFrame
train7_3hr


# In[ ]:


# Specify the start and end datetime values
start_datetime = pd.to_datetime('1/1/2021 0:00')
end_datetime = pd.to_datetime('12/31/2021 21:00')

# Create a new DataFrame with rows within the specified date range
test7_3hr = threehour_df[(threehour_df['DateTime'] >= start_datetime) & (threehour_df['DateTime'] <= end_datetime)].copy()

# Display the resulting DataFrame
test7_3hr


# # Dividing the datasets between the inputs and the target
# 

# ## Be careful here! Be sure that only one of these cells are active to run Cycle 1, 2, 3, 4, 5, 6, or 7. They are currently commented until you would like to run the other cycles.

# In[8]:


#Cycle 1 Data Splits for Model Runs
X_train = train1_3hr.iloc[:,2:-1].values.astype(float) 
y_train = train1_3hr.iloc[:,-1].values.astype(float)
    
X_test = test1_3hr.iloc[:,2:-1].values.astype(float) 
y_test = test1_3hr.iloc[:,-1].values.astype(float) 
    
X_val = validation1_3hr.iloc[:,2:-1].values.astype(float) 
y_val = validation1_3hr.iloc[:,-1].values.astype(float)
X_train


# In[ ]:


#Cycle 2 Data Splits for Model Runs
X_train = train2_3hr.iloc[:,1:-1].values.astype(float) 
y_train = train2_3hr.iloc[:,-1].values.astype(float)
    
X_test = test1_2hr.iloc[:,1:-1].values.astype(float) 
y_test = test1_2hr.iloc[:,-1].values.astype(float) 
    
X_val = validation2_3hr.iloc[:,1:-1].values.astype(float) 
y_val = validation2_3hr.iloc[:,-1].values.astype(float)
X_train


# In[ ]:


#Cycle 3 Data Splits for Model Runs
X_train = train3_3hr.iloc[:,1:-1].values.astype(float) 
y_train = train3_3hr.iloc[:,-1].values.astype(float)
    
X_test = test3_3hr.iloc[:,1:-1].values.astype(float) 
y_test = test3_3hr.iloc[:,-1].values.astype(float) 
    
X_val = validation3_3hr.iloc[:,1:-1].values.astype(float) 
y_val = validation3_3hr.iloc[:,-1].values.astype(float)
X_train


# In[ ]:


#Cycle 4 Data Splits for Model Runs
X_train = train4_3hr.iloc[:,1:-1].values.astype(float) 
y_train = train4_3hr.iloc[:,-1].values.astype(float)
    
X_test = test4_3hr.iloc[:,1:-1].values.astype(float) 
y_test = test4_3hr.iloc[:,-1].values.astype(float) 
    
X_val = validation4_3hr.iloc[:,1:-1].values.astype(float) 
y_val = validation4_3hr.iloc[:,-1].values.astype(float)
X_train


# In[ ]:


#Cycle 5 Data Splits for Model Runs
X_train = train5_3hr.iloc[:,1:-1].values.astype(float) 
y_train = train5_3hr.iloc[:,-1].values.astype(float)
    
X_test = test1_5hr.iloc[:,1:-1].values.astype(float) 
y_test = test1_5hr.iloc[:,-1].values.astype(float) 
    
X_val = validation5_3hr.iloc[:,1:-1].values.astype(float) 
y_val = validation5_3hr.iloc[:,-1].values.astype(float)
X_train


# In[ ]:


#Cycle 6 Data Splits for Model Runs
X_train = train6_3hr.iloc[:,1:-1].values.astype(float) 
y_train = train6_3hr.iloc[:,-1].values.astype(float)
    
X_test = test1_6hr.iloc[:,1:-1].values.astype(float) 
y_test = test1_6hr.iloc[:,-1].values.astype(float) 
    
X_val = validation6_3hr.iloc[:,1:-1].values.astype(float) 
y_val = validation6_3hr.iloc[:,-1].values.astype(float)
X_train


# In[ ]:


#Cycle 7 Data Splits for Model Runs
X_train = train7_3hr.iloc[:,1:-1].values.astype(float) 
y_train = train7_3hr.iloc[:,-1].values.astype(float)
    
X_test = test1_7hr.iloc[:,1:-1].values.astype(float) 
y_test = test1_7hr.iloc[:,-1].values.astype(float) 
    
X_val = validation7_3hr.iloc[:,1:-1].values.astype(float) 
y_val = validation7_3hr.iloc[:,-1].values.astype(float)
X_train


# # Visuals of Training, Validation, Testing

# # Model Creation

# In[9]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.callbacks import EarlyStopping
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor


# In[18]:


# To output the column names from a data frame
# df.columns

#BE AWARE! This will need to be changed depending on which dataset you are using
feature_names = ['precip_pred', 'cloud_base_height', 'zero_deg_level', 'relative_humidity',
                 'surface_pres', 'u_wind_10m', 'v_wind_10m', 'convective_ape', 'convective_inhib',
                 'CRP_summed1', 'RAS_summed1', 'RKP_summed1', 'precip_pred_3hrback', 'precip_pred_6hrback',
                 'precip_pred_9hrback', 'CRP_summed1_3hrback', 'CRP_summed1_6hrback', 'CRP_summed1_9hrback']


# Storage for all repetitions
all_metrics = []
all_predictions = []
all_importances = []

# For loop for 15 repetitions
for repetition in range(15):
    print(f"Training repetition {repetition + 1}...")

    # Train a new model in each repetition
    random_forest_out_of_bag = RandomForestRegressor(oob_score=True)
    random_forest_out_of_bag.fit(X_train, y_train)

    # Make predictions
    predicted_labels = random_forest_out_of_bag.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, predicted_labels)
    mse = mean_squared_error(y_test, predicted_labels)
    rmse = mean_squared_error(y_test, predicted_labels, squared=False)
    me = np.mean(y_test - predicted_labels)
    r2 = r2_score(y_test, predicted_labels)

    # Feature importance
    importances = list(random_forest_out_of_bag.feature_importances_)

    # Append metrics and predictions to storage
    all_metrics.append({'Repetition': repetition + 1, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'ME': me, 'R2': r2,
                        'OOB_Score': random_forest_out_of_bag.oob_score_})
    all_predictions.append(pd.Series(predicted_labels, name=f'Repetition_{repetition + 1}'))
    all_importances.append(pd.Series(importances, index=feature_names, name=f'Repetition_{repetition + 1}'))

    print(f"Training repetition {repetition + 1} complete.\n")

# Save metrics to a CSV file
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv('metrics.csv', index=False)

# Save predictions to a CSV file
predictions_df = pd.concat(all_predictions, axis=1)
predictions_df.to_csv('predictions.csv', index=False)

# Save feature importances to a CSV file
importances_df = pd.concat(all_importances, axis=1)
importances_df.to_csv('featureimportances.csv', index=True)


# In[ ]:


#Used to locate current directory in case you didn't save the predictions and the model evals in a local folder
import os
print("Current Working Directory:", os.getcwd())

