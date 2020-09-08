#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


# ## Loading Data

# In[2]:


sms_df = pd.read_csv("E:\\pooja\\DS\\ExcelR\\DataSets\\sms_raw_NB.csv",encoding = "ISO-8859-1")


# In[3]:


sms_df.head()


# ## Data Pre Processing

# In[4]:


import re


# In[6]:


stop_words = []


# In[7]:


with open("E:\\pooja\\DS\\ExcelR\\DataSets\\stop.txt") as stop_file:
    stop_words = stop_file.read()


# In[8]:


stop_words=stop_words.split('\n')   


# In[53]:


stop_words


# In[10]:


def data_cleansing(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))


# In[12]:


sms_df.text = sms_df.text.apply(data_cleansing)


# In[55]:


sms_df.shape
sms_df = sms_df.loc[sms_df.text != " ",:]
def split_into_words(i):
    return [word for word in i.split(" ")]


# ## Diving data into training and test set

# In[56]:


from sklearn.model_selection import train_test_split


# In[15]:


sms_train,sms_test = train_test_split(sms_df,test_size=0.3)


# ## Preparing word count matrix for train and test data

# In[17]:


sms_bow = CountVectorizer(analyzer=split_into_words).fit(sms_df.text)


# In[18]:


all_sms_matrix = sms_bow.transform(sms_df.text)


# In[19]:


all_sms_matrix


# In[20]:


train_sms_matrix = sms_bow.transform(sms_train.text)


# In[21]:


train_sms_matrix.shape


# In[22]:


test_sms_matrix = sms_bow.transform(sms_test.text)


# In[23]:


test_sms_matrix.shape


# ## Building the Model without TDIF matrices
# 

# In[24]:


from sklearn.naive_bayes import MultinomialNB, GaussianNB


# In[26]:


sms_mb = MultinomialNB()
sms_gb = GaussianNB()


# In[27]:


sms_mb.fit(train_sms_matrix,sms_train.type)


# In[28]:


train_pred_mb = sms_mb.predict(train_sms_matrix)


# In[29]:


accuracy_train_mb = np.mean(train_pred_mb==sms_train.type)


# In[30]:


accuracy_train_mb


# In[31]:


test_pred_mb = sms_mb.predict(test_sms_matrix)


# In[32]:


accuracy_test_mb = np.mean(test_pred_mb==sms_test.type)


# In[33]:


accuracy_test_mb


# In[34]:


sms_gb.fit(train_sms_matrix.toarray(),sms_train.type.values) 


# In[35]:


train_pred_gb = sms_gb.predict(train_sms_matrix.toarray())


# In[36]:


accuracy_train_gb = np.mean(train_pred_gb==sms_train.type)


# In[37]:


accuracy_train_gb


# In[39]:


test_pred_gb = sms_gb.predict(test_sms_matrix.toarray())


# In[57]:


accuracy_test_gb = np.mean(test_pred_gb==sms_test.type)


# In[58]:


accuracy_test_gb


# ## Building Model with TDIF matrices

# In[42]:


tfidf_transformer = TfidfTransformer().fit(all_sms_matrix)


# In[43]:


train_tfidf = tfidf_transformer.transform(train_sms_matrix)


# In[44]:


train_tfidf.shape


# In[45]:


test_tfidf = tfidf_transformer.transform(test_sms_matrix)


# In[46]:


test_tfidf.shape


# In[47]:


sms_mb.fit(train_tfidf,sms_train.type)
train_pred_mb = sms_mb.predict(train_tfidf)
accuracy_train_mb = np.mean(train_pred_mb==sms_train.type)


# In[48]:


accuracy_train_mb


# In[49]:


sms_gb.fit(train_tfidf.toarray(),sms_train.type)
train_pred_gb = sms_gb.predict(train_tfidf.toarray())
accuracy_train_gb = np.mean(train_pred_gb==sms_train.type)


# In[50]:


accuracy_train_gb


# In[51]:


test_pred_gb = sms_gb.predict(test_tfidf.toarray())
accuracy_test_gb = np.mean(test_pred_gb==sms_test.type)


# In[52]:


accuracy_test_gb


# In[ ]:




