#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ## Loading the data

# In[2]:


startup = pd.read_csv("E:\\pooja\\DS\\ExcelR\\DataSets\\50_Startups.csv")


# In[3]:


startup.head()


# ## EDA

# In[4]:


sns.pairplot(startup)


# In[5]:


sns.distplot(startup['Profit'])


# In[6]:


sns.boxplot(x='Profit', data=startup)


# In[8]:


sns.barplot(x='State', y='Profit', data=startup)


# ## Data Pre-processing

# In[5]:


startup_1 = startup


# In[6]:


startup_1.drop("State", axis=1, inplace=True)


# In[7]:


startup_1.head()


# ## Dividing data set into training and testing set

# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X = startup_1.drop("Profit", axis=1)
y = startup_1['Profit']


# In[10]:


X.head()


# In[11]:


y.head()


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# ## Featuring Scaling

# In[13]:


from sklearn.preprocessing import StandardScaler


# In[14]:


scaler = StandardScaler()


# In[15]:


scaler.fit(X_train)


# In[16]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ## Building the model

# In[17]:


from sklearn.neural_network import MLPRegressor


# In[18]:


startup_mlp = MLPRegressor(hidden_layer_sizes=(100,100,100,100), max_iter=1000)


# In[19]:


startup_mlp.fit(X_train, y_train)


# ## Predicting the values for test data 

# In[20]:


startup_prediction = startup_mlp.predict(X_test)


# In[21]:


startup_prediction


# In[22]:


y_test


# ## Finding the accuracy of model

# In[23]:


from sklearn import metrics


# In[24]:


print(metrics.r2_score(y_test, startup_prediction));


# In[25]:


print(metrics.mean_squared_log_error(y_test, startup_prediction))


# In[28]:


import math
print(math.sqrt(metrics.mean_squared_log_error(y_test, startup_prediction)))


# In[29]:


plt.figure(figsize=(10,10))
sns.regplot(y_test, startup_prediction, fit_reg=True, scatter_kws={"s": 100})


# In[ ]:




