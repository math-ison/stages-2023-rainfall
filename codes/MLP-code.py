#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# # Read in Datasets

# ## Within this workflow, please comment out all other dataframes when running one set of models.

# In[3]:


#Adjust to your local directories
threehour_df = pd.read_csv('3hr_precip_corrected_wlags.csv')
threehour_df = pd.read_csv('3hr_precip_corrected_wenv.csv')
threehour_df = pd.read_csv('3hr_precip_corrected_wenv_wmonth.csv')
threehour_df = pd.read_csv('3hr_precip_corrected_woenv_wlags.csv')


# In[4]:


threehour_df['DateTime'] = pd.to_datetime(threehour_df['DateTime'])


# ## Cycle 1 Data Splits 

# In[5]:


# Specify the start and end datetime values
start_datetime = pd.to_datetime('1/1/2021 0:00')
end_datetime = pd.to_datetime('12/31/2021 21:00')

# Create a new DataFrame with rows within the specified date range
train1_3hr = threehour_df[(threehour_df['DateTime'] >= start_datetime) & (threehour_df['DateTime'] <= end_datetime)].copy()

# Display the resulting DataFrame
train1_3hr


# In[6]:


# Specify the datetime value
target_datetime = pd.to_datetime('12/31/2021 21:00')

# Create a new DataFrame with rows up to the specified datetime
validation1_3hr = threehour_df[threehour_df['DateTime'] <= target_datetime].copy()
validation1_3hr


# In[7]:


# Specify the start and end datetime values
start_datetime = pd.to_datetime('1/1/2022 0:00')
end_datetime = pd.to_datetime('12/27/2022 21:00')

# Create a new DataFrame with rows within the specified date range
test1_3hr = threehour_df[(threehour_df['DateTime'] >= start_datetime) & (threehour_df['DateTime'] <= end_datetime)].copy()

# Display the resulting DataFrame
test1_3hr


# ## Cycle 2 Data Splits

# In[ ]:


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
target_datetime = pd.to_datetime('12/31/2021 21:00')

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


# # Dividing the datasets between the inputs and the target
# 

# ## Be careful here! Be sure that only one of these cells are active to run Cycle 1, 2, 3, or 4. They are currently commented until you would like to run the other cycles.

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
X_train = train2_3hr.iloc[:,2:-1].values.astype(float) 
y_train = train2_3hr.iloc[:,-1].values.astype(float)
    
X_test = test2_3hr.iloc[:,2:-1].values.astype(float) 
y_test = test2_3hr.iloc[:,-1].values.astype(float) 
    
X_val = validation2_3hr.iloc[:,2:-1].values.astype(float) 
y_val = validation2_3hr.iloc[:,-1].values.astype(float)
X_train


# In[ ]:


#Cycle 3 Data Splits for Model Runs
X_train = train3_3hr.iloc[:,2:-1].values.astype(float) 
y_train = train3_3hr.iloc[:,-1].values.astype(float)
    
X_test = test3_3hr.iloc[:,2:-1].values.astype(float) 
y_test = test3_3hr.iloc[:,-1].values.astype(float) 
    
X_val = validation3_3hr.iloc[:,2:-1].values.astype(float) 
y_val = validation3_3hr.iloc[:,-1].values.astype(float)
X_train


# In[ ]:


#Cycle 4 Data Splits for Model Runs
X_train = train4_3hr.iloc[:,2:-1].values.astype(float) 
y_train = train4_3hr.iloc[:,-1].values.astype(float)
    
X_test = test1_4hr.iloc[:,2:-1].values.astype(float) 
y_test = test1_4hr.iloc[:,-1].values.astype(float) 
    
X_val = validation4_3hr.iloc[:,2:-1].values.astype(float) 
y_val = validation4_3hr.iloc[:,-1].values.astype(float)
X_train


# # Hyperparameter Tuning (can skip because it has already been tuned)

# In[ ]:


import kerastuner as kt
from tensorflow import keras
from tensorflow.keras import layers

# Assuming you have already loaded and preprocessed your data
# Replace this with your actual data loading and preprocessing code
# For example, if you have DataFrames named train1_6hr, test1_6hr, and validation1_6hr:
# X_train = train1_6hr.iloc[:, 2:-1].values.astype(float)
# y_train = train1_6hr.iloc[:, -1].values.astype(float)
# X_test = test1_6hr.iloc[:, 2:-1].values.astype(float)
# y_test = test1_6hr.iloc[:, -1].values.astype(float)
# X_val = validation1_6hr.iloc[:, 2:-1].values.astype(float)
# y_val = validation1_6hr.iloc[:, -1].values.astype(float)

# Define the model-building function with activation functions as hyperparameters
def build_model(hp):
    model = keras.Sequential()
    
    # Input layer with dynamic input_shape
    model.add(layers.Dense(units=hp.Int('units', min_value=8, max_value=64, step=8),
                           activation=hp.Choice('input_activation', values=['relu', 'sigmoid']),
                           input_shape=X_train[0].shape))
    
    # Hidden layer
    model.add(layers.Dense(units=hp.Int('units', min_value=8, max_value=64, step=8),
                           activation=hp.Choice('hidden_activation', values=['relu', 'sigmoid'])))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Choose a loss function as a hyperparameter
    loss_function = hp.Choice('loss_function', values=['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error'])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss=loss_function,
                  metrics=['accuracy'])
    return model

# Instantiate the Hyperband tuner
tuner = kt.tuners.Hyperband(build_model,
                            objective='val_accuracy',
                            max_epochs=10,
                            factor=3,
                            directory='my_tuning_dir',
                            project_name='my_tuning_project')

# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=1000, validation_data=(X_val, y_val))

# Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters(1)[0]

# Build the final model with the best hyperparameters
final_model = build_model(best_hp)

# Train the final model on the full training data
final_model.fit(X_train, y_train, epochs=1000, validation_data=(X_val, y_val))

# Evaluate the final model on the test data
test_loss, test_acc = final_model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')

# Print the best hyperparameters
print('\nBest Hyperparameters:')
for key, value in best_hp.values.items():
    print(f'{key}: {value}')


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
from keras.layers import Dense
from keras.layers import BatchNormalization


# In[10]:


# Assuming you have a list of feature names
feature_names = ['precip_pred','cloud_base_height','zero_deg_level','relative_humidity','surface_pres','u_wind_10m','v_wind_10m','convective_ape','convective_inhib', 'CRP_summed1','RAS_summed1','RKP_summed1']  # Replace with your actual feature names
("#'precip_pred_3hrback','precip_pred_6hrback','precip_pred_9hrback','CRP_summed1_3hrback','CRP_summed1_6hrback','CRP_summed1_9hrback'")
# Storage for all repetitions
all_metrics = []
all_predictions = []
all_importances = []



# Define a new model for each repetition
model = Sequential()
model.add(Dense(units=8, input_shape=X_train[0].shape, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    
# For loop for 15 repetitions
for repetition in range(15):
    print(f"Training repetition {repetition + 1}...")    
    
    model.fit(X_train, y_train, epochs=10000, batch_size=128, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)

    # Make predictions on the test set
    predicted_labels = model.predict(X_test).flatten()
 
    # Evaluation metrics
    mae = mean_absolute_error(y_test, predicted_labels)
    mse = mean_squared_error(y_test, predicted_labels)
    rmse = np.sqrt(mse)
    me = np.mean(y_test - predicted_labels)
    r2 = r2_score(y_test, predicted_labels)

    # Permutation feature importance
    importances = []
    for feature in range(X_test.shape[1]):
        permuted_X = X_test.copy()
        np.random.shuffle(permuted_X[:, feature])
        permuted_predictions = model.predict(permuted_X).flatten()
        permuted_mse = mean_squared_error(y_test, permuted_predictions)
        importances.append(mse - permuted_mse)

    # Append metrics, predictions, and importances to storage
    all_metrics.append({'Repetition': repetition + 1, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'ME': me, 'R2': r2})
    all_predictions.append(pd.Series(predicted_labels, name=f'Repetition_{repetition + 1}'))
    all_importances.append(pd.Series(importances, index=feature_names, name=f'Repetition_{repetition + 1}'))

    print(f"Training repetition {repetition + 1} complete.\n")
    
    model.summary()

# Save metrics to a CSV file
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv('metrics.csv', index=False)

# Save predictions to a CSV file
predictions_df = pd.concat(all_predictions, axis=1)
predictions_df.to_csv('predictions.csv', index=False)

# Save feature importances to a CSV file
importances_df = pd.concat(all_importances, axis=1)
importances_df.to_csv('feature_importances.csv', index=True)


# In[ ]:


#Used to locate current directory in case you didn't save the predictions and the model evals in a local folder
import os
print("Current Working Directory:", os.getcwd())

