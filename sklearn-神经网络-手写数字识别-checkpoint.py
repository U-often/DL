#!/usr/bin/env python
# coding: utf-8

# ![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/README_IMG/01.jpg)
# AI MOOC： **www.ai-xlab.com**  
# 如果你也是AI爱好者，可以添加我的微信一起交流：**sdxxqbf**

# In[1]:


# pip install scikit-learn --upgrade
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix


# In[2]:


digits = load_digits()#载入数据
x_data = digits.data #数据
y_data = digits.target #标签

# 标准化
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data) #分割数据1/4为测试数据，3/4为训练数据


# In[3]:


mlp = MLPClassifier(hidden_layer_sizes=(100,50) ,max_iter=500)
mlp.fit(x_train,y_train )


# In[4]:


predictions = mlp.predict(x_test)
print(classification_report(y_test, predictions))


# In[5]:


print(confusion_matrix(y_test,predictions))


# In[ ]:




