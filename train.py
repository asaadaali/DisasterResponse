#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[58]:


# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle


# In[59]:


# load data from database
engine = create_engine('sqlite:///DisasterResponse.db')
engine.table_names()


# In[60]:


df = pd.read_sql_table('Message', con=engine)
df.head()


# In[61]:


df.shape


# In[62]:


X = df.message 
Y = df.loc[:, 'related':'direct_report']


# In[63]:


print(X.shape) 
X.head()


# In[64]:


print(Y.shape)
Y.head()


# ### 2. Write a tokenization function to process your text data

# In[65]:


def tokenize(text):
    #case normalization to remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #tokenization methods 
    
    words = word_tokenize(text)
    
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]

    
    return words


# In[66]:


for message in X[:5]:
    tokens = tokenize(message)
    print(message)
    print(tokens, '\n')


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[67]:


cls = MultiOutputClassifier(DecisionTreeClassifier())

pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', cls)
    ])


# In[68]:


pipeline.get_params()


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[69]:


X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size = 0.2, random_state = 45)


# In[70]:


pipeline.fit(X_train, y_train)


# In[71]:


y_pred = pipeline.predict(X_test)


# In[72]:


y_pred


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[73]:


for i, col in enumerate(Y.columns):
    print(f'-----------------------{i, col}----------------------------------')
    
#     y_actual = list(y_test.values[:, i])
#     y_pred = list(y_pred[:, i])
    print(classification_report(list(y_test.values[:, i]), list(y_pred[:, i])))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[74]:


pipeline.get_params()


# In[75]:


parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}

cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=2, verbose=3)
cv.fit(X_train, y_train)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[76]:


y_pred = cv.predict(X_test)


# In[77]:


for i, col in enumerate(Y.columns):
    print(f'-----------------------{i, col}----------------------------------')
    
#     y_actual = list(y_test.values[:, i])
#     y_pred = list(y_pred[:, i])
    print(classification_report(list(y_test.values[:, i]), list(y_pred[:, i])))


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[78]:


moc = MultiOutputClassifier(RandomForestClassifier())

pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', moc)
    ])


# In[79]:


X_train, X_test, y_train, y_test = train_test_split(X, Y)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


# In[80]:


for i, col in enumerate(Y.columns):
    print(f'-----------------------{i, col}----------------------------------')
    
#     y_actual = list(y_test.values[:, i])
#     y_pred = list(y_pred[:, i])
    print(classification_report(list(y_test.values[:, i]), list(y_pred[:, i])))


# ### 9. Export your model as a pickle file

# In[ ]:


pickle.dump(cv, open('model.pkl', 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




