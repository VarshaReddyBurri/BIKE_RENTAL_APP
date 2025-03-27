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


pip install xgboost


# In[3]:


data = pd.read_csv('Bike_Rental_Data.csv')
data


# In[ ]:





# In[4]:


data.info()


# In[5]:


features_categorical= [
   'holiday', 'workingday', 'season','weathersit'
]


# In[6]:


#converting the categorical to continous
data=pd.get_dummies(data)
data


# In[7]:


data.info()


# In[8]:


#r Replacing the columns with "_" which contains space between the column names
data.columns = data.columns.str.replace(" ", "_")
data


# In[9]:


# Define features and target variable
X = data.drop(columns=['cnt'])  # Assuming 'cnt' is the target variable
y = data['cnt']


# In[10]:


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[12]:


# Make predictions
y_pred_rf = rf_model.predict(X_test)


# In[13]:


# Make predictions on training
y_pred_rf1 = rf_model.predict(X_train)


# In[14]:


# Evaluate model performance on training
mse_rf1 = mean_squared_error(y_train, y_pred_rf1)
r2_rf1 = r2_score(y_train, y_pred_rf1)


# In[15]:


print(f"Random Forest - Mean Squared Error: {mse_rf1:.2f}")
print(f"Random Forest - R² Score: {r2_rf1:.2f}")


# In[16]:


# Evaluate model performance on testing
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)


# In[17]:


print(f"Random Forest - Mean Squared Error: {mse_rf:.2f}")
print(f"Random Forest - R² Score: {r2_rf:.2f}")


# In[18]:


plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Bike Rentals")


# In[19]:


import xgboost as xgb


# In[20]:


xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)


# In[21]:


xg_reg


# In[22]:


xg_reg.fit(X_train, y_train)


# In[23]:


predictions = xg_reg.predict(X_test)


# In[24]:


mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)


# In[25]:


r2 = r2_score(y_test, predictions)
print("R-squared (R²):", r2)


# In[26]:


plt.scatter(y_test, predictions)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Bike Rentals")


# In[27]:


# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save the scaler
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and Scaler saved successfully!")


# In[ ]:




