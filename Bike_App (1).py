#!/usr/bin/env python
# coding: utf-8

# In[114]:


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
import xgboost as xgb


# In[ ]:





# In[117]:


data = pd.read_csv('Bike_Rental_Data.csv')
data


# In[ ]:





# In[120]:


data.info()


# In[122]:


features_categorical= [
   'holiday', 'workingday', 'season','weathersit'
]


# In[124]:


#converting the categorical to continous
data=pd.get_dummies(data)
data


# In[126]:


data.info()


# In[128]:


#r Replacing the columns with "_" which contains space between the column names
data.columns = data.columns.str.replace(" ", "_")
data


# In[130]:


# Define features and target variable
X = data.drop(columns=['cnt'])  # Assuming 'cnt' is the target variable
y = data['cnt']


# In[132]:


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[134]:


# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[135]:


# Make predictions
y_pred_rf = rf_model.predict(X_test)


# In[136]:


# Make predictions on training
y_pred_rf1 = rf_model.predict(X_train)


# In[137]:


# Evaluate model performance on training
mse_rf1 = mean_squared_error(y_train, y_pred_rf1)
r2_rf1 = r2_score(y_train, y_pred_rf1)


# In[138]:


print(f"Random Forest - Mean Squared Error: {mse_rf1:.2f}")
print(f"Random Forest - R² Score: {r2_rf1:.2f}")


# In[139]:


# Evaluate model performance on testing
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)


# In[140]:


print(f"Random Forest - Mean Squared Error: {mse_rf:.2f}")
print(f"Random Forest - R² Score: {r2_rf:.2f}")


# In[141]:


plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Bike Rentals")


# In[142]:


import xgboost as xgb


# In[143]:


xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)


# In[144]:


xg_reg


# In[145]:


xg_reg.fit(X_train, y_train)


# In[146]:


predictions = xg_reg.predict(X_test)


# In[147]:


mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)


# In[148]:


r2 = r2_score(y_test, predictions)
print("R-squared (R²):", r2)


# In[149]:


plt.scatter(y_test, predictions)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Bike Rentals")


# In[150]:


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




