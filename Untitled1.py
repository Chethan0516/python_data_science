#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv("C:/Users/Space/Downloads/archive (4)/tesla.csv")


# In[5]:


df


# In[20]:


dff = df.iloc[0,::]


# In[16]:


dff


# In[22]:


d = pd.DataFrame(dff).transpose()
d


# In[5]:


df.isna().value_counts()


# In[6]:


cormap = df.corr()
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cormap, annot = True)


# In[7]:


def get_corelated_col(cor_dat, threshold): 
  # Cor_data to be column along which corelation to be measured 
  #Threshold be the value above which of corelation to considered
  feature=[]
  value=[]

  for i ,index in enumerate(cor_dat.index):
    if abs(cor_dat[index]) > threshold:
      feature.append(index)
      value.append(cor_dat[index])

  df = pd.DataFrame(data = value, index = feature, columns=['corr value'])
  return df


# In[8]:


top_corelated_values = get_corelated_col(cormap['Close'], 0.60)
top_corelated_values


# In[9]:


df = df[top_corelated_values.index]
df.head()


# In[10]:


sns.pairplot(df)
plt.tight_layout()


# In[11]:


X = df.drop(['Close'], axis=1)
y = df['Close']


# In[12]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X.head()


# In[26]:


#now lets split data in test train pairs

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle=False)

Acc = []


# # Linear

# In[27]:


from sklearn.linear_model import LinearRegression

# model training

model_1 = LinearRegression()
model_1.fit(X_train, y_train)


# In[28]:


# prediction
y_pred_1 = model_1.predict(X_test)
pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_1})
pred_df.head()


# In[ ]:





# In[29]:


# Measure the Accuracy Score

from sklearn.metrics import r2_score
 
print("Accuracy score of the predictions: {0}".format(r2_score(y_test, y_pred_1)))
Acc.append(r2_score(y_test, y_pred_1))


# In[30]:


plt.figure(figsize=(8,8))
plt.ylabel('Close Price', fontsize=16)
plt.plot(pred_df)
plt.legend(['Actual Value', 'Predictions'])
plt.show()


# In[ ]:





# In[32]:


df


# In[ ]:




