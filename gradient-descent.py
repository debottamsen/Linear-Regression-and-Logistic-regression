#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_classification
import numpy as np
X, y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=20)


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)


# In[4]:


from sklearn.linear_model import LogisticRegression
lor = LogisticRegression(penalty='none',solver='sag')
lor.fit(X,y)


# In[5]:


print(lor.coef_)
print(lor.intercept_)


# In[6]:


m1 = -(lor.coef_[0][0]/lor.coef_[0][1])
b1 = -(lor.intercept_/lor.coef_[0][1])


# In[7]:


x_input = np.linspace(-3,3,100)
y_input = m1*x_input + b1


# In[8]:


def gd(X,y):
    
    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.5
    
    for i in range(5000):
        y_hat = sigmoid(np.dot(X,weights))
        weights = weights + lr*(np.dot((y-y_hat),X)/X.shape[0])
        
    return weights[1:],weights[0]
        


# In[9]:


def sigmoid(z):
    return 1/(1 + np.exp(-z))


# In[10]:


coef_,intercept_ = gd(X,y)


# In[11]:


m = -(coef_[0]/coef_[1])
b = -(intercept_/coef_[1])


# In[12]:


x_input1 = np.linspace(-3,3,100)
y_input1 = m*x_input1 + b


# In[13]:


plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='red',linewidth=3)
plt.plot(x_input1,y_input1,color='black',linewidth=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,2)


# In[14]:


np.insert(X,0,1,axis=1)


# In[15]:


X1 = np.insert(X,0,1,axis=1)
np.ones(X1.shape[1])


# In[ ]:




