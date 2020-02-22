#!/usr/bin/env python
# coding: utf-8

# $$\small \color{green}{\textbf{The Threshold ARCH (TARCH) Modelling}}$$ 
# 
# 
# $$\large \color{blue}{\textbf{Phuong Van Nguyen}}$$
# $$\small \color{red}{\textbf{ phuong.nguyen@summer.barcelonagse.eu}}$$
# 
# 
# This computer program was written by Phuong V. Nguyen, based on the $\textbf{Anacoda 1.9.7}$ and $\textbf{Python 3.7}$.
# 
# $\text{2. Dataset:}$ 
# 
# One can download the dataset used to replicate my project at my Repositories on the Github site below
# 
# https://github.com/phuongvnguyen/TARCH-Modelling/blob/master/data.xlsx
# 
# # Preparing Problem
# 
# ##  Loading Libraries

# In[1]:


import warnings
import itertools
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
#import pmdarima as pm
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from arch import arch_model
from arch.univariate import GARCH


# ## Defining some varibales for printing the result

# In[2]:


Purple= '\033[95m'
Cyan= '\033[96m'
Darkcyan= '\033[36m'
Blue = '\033[94m'
Green = '\033[92m'
Yellow = '\033[93m'
Red = '\033[91m'
Bold = "\033[1m"
Reset = "\033[0;0m"
Underline= '\033[4m'
End = '\033[0m'


# ##  Loading Dataset

# In[3]:


data = pd.read_excel("data.xlsx")


# # Data Exploration and Preration
# 
# ## Data exploration

# In[4]:


data.head(5)


# ## Computing returns
# ### Picking up the close prices

# In[5]:


closePrice = data[['DATE','CLOSE']]
closePrice.head(5)


# ### Computing the daily returns

# In[6]:


closePrice['Return'] = closePrice['CLOSE'].pct_change()
closePrice.head()


# In[7]:


daily_return=closePrice[['DATE','Return']]
daily_return.head()


# ### Reseting index

# In[8]:


daily_return =daily_return.set_index('DATE')
daily_return.head()


# In[9]:


daily_return = 100 * daily_return.dropna()
daily_return.head()


# In[10]:


daily_return.index


# ### Plotting returns

# In[12]:


sns.set()
fig=plt.figure(figsize=(12,7))
plt.plot(daily_return.Return['2007':'2013'],LineWidth=1)
plt.autoscale(enable=True,axis='both',tight=True)
#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The Log Daily Returns', fontsize=18,fontweight='bold')
plt.title('19/09/2007- 31/12/2014',fontsize=15,fontweight='bold',color='k')
plt.ylabel('Return (%)',fontsize=17)
plt.xlabel('Source: The Daily Close Price-based Calculations',fontsize=17,fontweight='normal',color='k')


# # Modelling TARCH model
# 
# $$\text{Mean equation:}$$
# $$r_{t}=\mu + \epsilon_{t}$$
# 
# $$\text{Conditional Volatility (Variance) equation:}$$
# $$\sigma^{k}_{t}= \omega + \alpha |\epsilon_{t}|^{k} + \gamma |\epsilon_{t-1}|^{k} \mathbf{I}_{[\epsilon_{t-1}<0]}+\beta\sigma^{k}_{t-1}$$
# 
# $$\text{where:}$$
# 
# 
# $$\epsilon_{t}= \sigma_{t} e_{t}$$
# 
# $$e_{t} \sim N(0,1)$$
# 
# $$\mathbf{I} $$ $$\text{is an indicator function that takes the value 1 when its argument is true}$$
# 

# In[26]:


tarch= arch_model(daily_return, p=1, o=1, q=1, power=1.0)
results = tarch.fit(update_freq=1, disp='on')
print(results.summary())


# # Checking the residual
# ## Checking residuals

# In[23]:


fig=plt.figure(figsize=(10,5))
plt.plot(results.resid['2007':'2013'],LineWidth=1)
plt.autoscale(enable=True,axis='both',tight=True)
#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The Residuals of Mean Equation', 
             fontsize=15,fontweight='bold',
            color='b')
plt.title('19/09/2007- 31/12/2013',fontsize=10,
          fontweight='bold',color='b')
plt.ylabel('Residuals',fontsize=10,color='k')


# ## Conditional volatility

# In[24]:


fig=plt.figure(figsize=(10,5))
plt.plot(results.conditional_volatility['2007':'2013'],LineWidth=1)
plt.autoscale(enable=True,axis='both',tight=True)
#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The TARCH-based Conditional Volatilities', 
             fontsize=15,fontweight='bold',
            color='b')
plt.title('19/09/2007- 31/12/2013',fontsize=10,
          fontweight='bold',color='b')
plt.ylabel('Volatilies',fontsize=10,color='b')


# ## Residuals standardized by conditional volatility

# In[25]:


fig=plt.figure(figsize=(10,5))
plt.plot(results.std_resid['2007':'2013'],LineWidth=1)
plt.autoscale(enable=True,axis='both',tight=True)
#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The TARCH-based Residuals standardized by conditional volatility', 
             fontsize=15,fontweight='bold',
            color='b')
plt.title('19/09/2007- 31/12/2013',fontsize=10,
          fontweight='bold',color='b')
plt.ylabel('Volatilies',fontsize=10,color='b')


# # ARCH LM test for conditional heteroskedasticity
# 
# Arch Lagrange Multiplier or ARCH LM tests whether coefficients in the regression:
# 
# 
# $$\epsilon^{2}_{t}=\alpha_{0} + \alpha_{1}\epsilon^{2}_{t-1}+ \alpha_{2}\epsilon^{2}_{t-2}+...++ \alpha_{p}\epsilon^{2}_{t-p}+e_{t} $$
# 
# where 
# 
# $$\epsilon_{t}=r_{t}-mean(r_{t}) $$
# 
# $H_{0}:$ $ \alpha_{0}=\alpha_{1}=...=\alpha_{p}$ or Residuals are homoskedastic/ No ARCH effect
# 

# In[17]:


print(results.arch_lm_test(lags=10))


# P-value is very big, we can not reject the null hypothesis
