#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd 


# In[2]:


import matplotlib.pyplot as plt
from scipy import stats

import seaborn as sns


# In[3]:


import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[4]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# In[5]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[8]:


wines_df = pd.read_csv("wineQT.csv")
wines_df.sample(frac=1).reset_index(drop=True)

wines_df.head()


# In[9]:


display(wines_df.isna().sum())


# In[10]:


correlation = wines_df.corr()

fig = plt.subplots(figsize=(10,10))
sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='Blues')


# In[11]:


wines_df_before = wines_df
wines_df_after = wines_df.drop(['fixed acidity','citric acid','density'], axis = 1)

X1 = sm.tools.add_constant(wines_df_before)
X2 = sm.tools.add_constant(wines_df_after)

series_before = pd.Series([variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])], index=X1.columns)
series_after = pd.Series([variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])], index=X2.columns)

display(series_before)
display(series_after)

wines_df = wines_df_after


# In[12]:


x = np.array(wines_df.loc[:, wines_df.columns != 'quality'])
y = np.array(wines_df['quality'])


# In[13]:


desc_df = wines_df.describe()

desc_df.loc['+3_std'] = desc_df.loc['mean'] + (desc_df.loc['std'] * 3)
desc_df.loc['-3_std'] = desc_df.loc['mean'] - (desc_df.loc['std'] * 3)

desc_df


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=1)

print(x_train.shape, x_test.shape)


# In[15]:


regression_model = LinearRegression()

regression_model.fit(x_train, y_train)


# In[16]:


y_predict = regression_model.predict(x_test)
y_predict = np.round(y_predict)

sum = 0
for i,n in enumerate(y_test):
    if n == y_predict[i]:
        sum += 1
print(sum/len(y_test))


# In[ ]:




