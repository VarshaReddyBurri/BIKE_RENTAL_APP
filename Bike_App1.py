#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import pickle  # Importing pickle


# In[2]:





# In[5]:


data = pd.read_csv('Bike_Rental_Data.csv')
data


# In[ ]:





# In[7]:


data.info()


# In[8]:


features_categorical= [
   'holiday', 'workingday', 'season','weathersit'
]


# In[9]:


#converting the categorical to continous
data=pd.get_dummies(data)
data


# In[10]:


data.info()


# In[11]:


#r Replacing the columns with "_" which contains space between the column names
data.columns = data.columns.str.replace(" ", "_")
data


# In[12]:


# Define features and target variable
X = data.drop(columns=['cnt'])  # Assuming 'cnt' is the target variable
y = data['cnt']


# In[13]:


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[15]:


# Make predictions
y_pred_rf = rf_model.predict(X_test)


# In[16]:


# Make predictions on training
y_pred_rf1 = rf_model.predict(X_train)


# In[17]:


# Evaluate model performance on training
mse_rf1 = mean_squared_error(y_train, y_pred_rf1)
r2_rf1 = r2_score(y_train, y_pred_rf1)


# In[18]:


print(f"Random Forest - Mean Squared Error: {mse_rf1:.2f}")
print(f"Random Forest - R² Score: {r2_rf1:.2f}")


# In[19]:


# Evaluate model performance on testing
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)


# In[20]:


print(f"Random Forest - Mean Squared Error: {mse_rf:.2f}")
print(f"Random Forest - R² Score: {r2_rf:.2f}")


# In[21]:


plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Bike Rentals")


# In[ ]:




