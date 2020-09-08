#!/usr/bin/env python
# coding: utf-8

# ## Loading libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Loading the data 

# In[2]:


salary_train = pd.read_csv("E:\\pooja\\DS\\ExcelR\\DataSets\\SalaryData_Train.csv")


# In[3]:


salary_test = pd.read_csv("E:\\pooja\\DS\\ExcelR\\DataSets\\SalaryData_Test.csv")


# In[5]:


salary_train.head()


# ## EDA

# In[6]:


sns.countplot(x="Salary", data=salary_train)


# In[7]:


sns.countplot(x="Salary", data=salary_train, hue="sex")


# In[9]:


sns.countplot(x="Salary", data=salary_train, hue="race")


# In[11]:


sns.distplot(salary_train["age"], kde=False)


# In[12]:


sns.countplot(x='Salary', data=salary_train, hue="workclass")


# ## Data Pre-Processing

# In[13]:


from sklearn import preprocessing


# In[14]:


categorical_columns = ["workclass", "education", "maritalstatus", "occupation", "relationship", "race", "sex", "native"]


# In[15]:


convert_to_numeric = preprocessing.LabelEncoder()


# In[16]:


for i in categorical_columns:
    salary_train[i] = convert_to_numeric.fit_transform(salary_train[i])
    salary_test[i] = convert_to_numeric.fit_transform(salary_test[i])


# In[17]:


salary_train.head()


# In[18]:


salary_test.head()


# In[19]:


sal_train_X = salary_train.drop("Salary", axis=1)
sal_train_Y = salary_train["Salary"]

sal_test_X = salary_test.drop("Salary", axis=1)
sal_test_Y = salary_test["Salary"]


# In[22]:


sal_train_X.head()


# ## Training the model on training data and predicting the values on test data 

# In[23]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix


# In[24]:


sal_gauss = GaussianNB()
sal_multi = MultinomialNB()


# In[25]:


sal_gauss_pred = sal_gauss.fit(sal_train_X, sal_train_Y).predict(sal_test_X)


# In[26]:


sal_multi_pred = sal_multi.fit(sal_train_X, sal_train_Y).predict(sal_test_X)


# ##  FInding the accuracy of model

# In[27]:


pd.crosstab(sal_test_Y, sal)


# In[28]:


pd.crosstab(sal_test_Y, sal_multi_pred)


# In[31]:


labels = np.unique(sal_test_Y)
a = confusion_matrix(sal_test_Y, sal_gauss_pred, labels=labels)
pd.DataFrame(a, index=labels, columns=labels)


# In[32]:


print("Accuracy of Guassian Model")
print((10759 + 1209) / (10759 + 1209 + 601 + 2491))


# In[33]:


pd.crosstab(sal_test_Y,sal_multi_pred)


# In[34]:


labels = np.unique(sal_test_Y)
a = confusion_matrix(sal_test_Y, sal_multi_pred, labels=labels)
pd.DataFrame(a, index=labels, columns=labels)


# In[35]:


print("Accuracy od Multinomial Model")
print((10891 + 780) / (10891 + 780 + 2920 + 469))


# In[ ]:




