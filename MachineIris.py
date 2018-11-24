#!/usr/bin/env python
# coding: utf-8

# # Welcome to Jupyter!

# This repo contains an introduction to [Jupyter](https://jupyter.org) and [IPython](https://ipython.org).
# 
# Outline of some basics:
# 
# * [Notebook Basics](../examples/Notebook/Notebook%20Basics.ipynb)
# * [IPython - beyond plain python](../examples/IPython%20Kernel/Beyond%20Plain%20Python.ipynb)
# * [Markdown Cells](../examples/Notebook/Working%20With%20Markdown%20Cells.ipynb)
# * [Rich Display System](../examples/IPython%20Kernel/Rich%20Output.ipynb)
# * [Custom Display logic](../examples/IPython%20Kernel/Custom%20Display%20Logic.ipynb)
# * [Running a Secure Public Notebook Server](../examples/Notebook/Running%20the%20Notebook%20Server.ipynb#Securing-the-notebook-server)
# * [How Jupyter works](../examples/Notebook/Multiple%20Languages%2C%20Frontends.ipynb) to run code in different languages.

# You can also get this tutorial and run it on your laptop:
# 
#     git clone https://github.com/ipython/ipython-in-depth
# 
# Install IPython and Jupyter:
# 
# with [conda](https://www.anaconda.com/download):
# 
#     conda install ipython jupyter
# 
# with pip:
# 
#     # first, always upgrade pip!
#     pip install --upgrade pip
#     pip install --upgrade ipython jupyter
# 
# Start the notebook in the tutorial directory:
# 
#     cd ipython-in-depth
#     jupyter notebook

# In[1]:


5-2


# In[2]:


import pandas

# In[3]:


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import numpy as np


# In[6]:


iris = load_iris()
iris.keys()


# In[8]:


print(iris['DESCR'][:193]+'\n...')


# In[9]:


iris['target_names']


# In[12]:


iris['target']
iris['feature_names']


# In[15]:


print(iris['data'].shape)
iris['data'][:10]


# In[26]:


x_train,x_test,y_train,y_test = train_test_split(iris['data'],iris['target'], random_state = 0)
print (x_train.shape)
print (x_test.shape)


# In[28]:


fig, ax = plt.subplots(3, 3, figsize=(15,15))

for i in range(3):
    for j in range(3):
        ax[i,j].scatter(x_train[:,j], x_train[:,i + 1], c=y_train, s=60)
        ax[i,j].set_xticks(())
        ax[i,j].set_yticks(())
        
        if i == 2:
            ax[i,j].set_xlabel(iris['feature_names'][j])
        if j == 0: 
            ax[i,j].set_ylabel(iris['feature_names'][i+1])
        if j > i:
            ax[i,j].set_visible(False)


# In[29]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)


# In[31]:


x_new = np.array([[5, 2.9, 1, 0.2]])
x_new.shape


# In[33]:


prediction = knn.predict(x_new)
prediction


# In[35]:


iris['target_names'][prediction]


# In[37]:


knn.score(x_test,y_test)


# In[ ]:




