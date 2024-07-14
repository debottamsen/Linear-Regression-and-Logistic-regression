#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df=pd.read_csv(r"C:\Users\Debottam.Sen\Downloads\placement.csv")


# In[4]:


df


# In[5]:


df.head()


# In[6]:


plt.scatter(df['cgpa'],df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')


# In[7]:


x=df.iloc[:,0:1]
y=df.iloc[:,-1]


# In[8]:


x


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[10]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[11]:


lr.fit(x_train,y_train)


# In[12]:


y_test


# In[13]:


lr.predict(x_test.iloc[1].values.reshape(1,1))


# In[14]:


plt.scatter(df['cgpa'],df['package'])
plt.plot(x_train,lr.predict(x_train),color='red')
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')


# In[15]:


m=lr.coef_


# In[16]:


m


# In[17]:


b=lr.intercept_


# In[18]:


b


# In[19]:


#y=mx+b
m

