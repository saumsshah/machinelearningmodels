#!/usr/bin/env python
# coding: utf-8

# In[1]:


#extracting the data

import numpy as np
import pandas as pd
import urllib
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
import xlrd 
import re

column_names = [ "Headline",  "Category"]
headlines = pd.DataFrame(columns = column_names)

for i in range(0,25):

    # Give the location of the file 
    loc = 'C:\Saumil\Data Science\Classification\Data2019.xlsx'
    # To open Workbook 
    headlines1 = pd.read_excel(loc,
    sheet_name=i,
    header=0,
    index_col=False,
    keep_default_na=True
    )

    headlines = headlines.append(headlines1,ignore_index=True)

headlines['label'] = headlines['Category'].apply(lambda x: 0 if x=='High' else (1 if x=='Medium' else 2))

label,counts = np.unique(headlines['Category'], return_counts = True)
print(dict(zip(label, counts)))


# In[2]:


#converting to lower case

headlines['Headline'] = headlines.apply(lambda x: x.astype(str).str.lower())

#removing special characters
spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–"]
for char in spec_chars:
    headlines['Headline'] = headlines['Headline'].str.replace(char, ' ')
    
#removing extra white space created

headlines['Headline'] = headlines['Headline'].str.split().str.join(" ")   

#converting numbers to num

headlines['Headline'] = headlines['Headline'].str.replace('\d+', 'num')

#removing single letters

headlines['Headline'] = headlines['Headline'].map(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))

headlines.tail()


# In[3]:


#lemmatization

import nltk

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


headlines['Headline'] = headlines['Headline'].apply(lemmatize_text).str.join(" ")


# In[4]:


# test train data. Run this code for unigrams

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(headlines['Headline'], headlines['label'] , test_size = 0.2 , random_state = 42)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words= "english")
X_train_cv = cv.fit_transform(X_train,y=None)
X_test_cv = cv.transform(X_test)

from sklearn.feature_extraction.text import TfidfTransformer
tf = TfidfTransformer().fit(X_train_cv)
X_train_tf = tf.fit_transform(X_train_cv)
X_test_tf = tf.fit_transform(X_test_cv)
X_train_tf.shape


# In[5]:


# test train data. Run this code for bi-grams

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(headlines['Headline'], headlines['label'] , test_size = 0.2, random_state = 42)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range =(2, 2))
X_train_cv = cv.fit_transform(X_train,y=None)
X_test_cv = cv.transform(X_test)

from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(ngram_range =(2, 2))
X_train_tv = tv.fit_transform(X_train)
X_test_tv = tv.transform(X_test)


from sklearn.feature_extraction.text import TfidfTransformer
tf = TfidfTransformer().fit(X_train_tv)
X_train_tf = tf.fit_transform(X_train_tv)
X_test_tf = tf.fit_transform(X_test_tv)
X_train_tf.shape


# In[6]:


#modeling and accuracy

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
model = LogisticRegression(penalty= 'l2', multi_class='auto')
model.fit(X_train_tf,y_train)
y_pred = model.predict(X_test_tf)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(model.score(X_test_tf, y_test)))


# In[7]:


#confusion matrix and individual accuracy

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

y_true = y_test
y_pred = y_pred
data = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size

high_acc = data[0][0] / (data[0][0] + data[0][1] + data[0][2])
low_acc = data[1][1] / (data[1][0] + data[1][1] + data[1][2])
med_acc = data[2][2] / (data[2][0] + data[2][1] + data[2][2])

print ("High accuracy is: " + str(high_acc)+  "   Low accuracy is: " + str(low_acc) + "   Medium accuracy is: " + str(med_acc) )


# In[ ]:


#Predicting individual data unigram

data = ["MI: The Daily Dose Latin America: Cielo's Q1 profit plunges; Paraguay GDP contraction seen"]
testHeadline = pd.DataFrame(data,columns = ['Headline'])
X_static_test = testHeadline['Headline']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words= "english")
X_train_cv = cv.fit_transform(X_train,y=None)
X_statictest_cv = cv.transform(X_static_test)


predictions_each = model.predict_proba(X_statictest_cv)
predictions1 = model.predict(X_statictest_cv)

final_statement = "High is " + np.array2string(predictions_each[0][0] * 100.0, precision=4)  + "%"  + "Medium is " + np.array2string(predictions_each[0][1] * 100.0, precision=4)  + "%" + "Low is " + np.array2string(predictions_each[0][2] * 100.0,precision=4)  + "%" 

print(final_statement)


# In[10]:


#Predicting individual data bi-gram

data = ["MI: The Daily Dose Latin America: Cielo's Q1 profit plunges; Paraguay GDP contraction seen"]
testHeadline = pd.DataFrame(data,columns = ['Headline'])
X_static_test = testHeadline['Headline']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range =(2, 2))
X_train_cv = cv.fit_transform(X_train,y=None)
X_statictest_cv = cv.transform(X_static_test)


predictions_each = model.predict_proba(X_statictest_cv)
predictions1 = model.predict(X_statictest_cv)

final_statement = "High is " + np.array2string(predictions_each[0][0] * 100.0, precision=4)  + "%"  + "Medium is " + np.array2string(predictions_each[0][1] * 100.0, precision=4)  + "%" + "Low is " + np.array2string(predictions_each[0][2] * 100.0,precision=4)  + "%" 

print(final_statement)


# In[12]:


#Plotting the graph for the output

import matplotlib.pyplot as plt  
predictions_each_series = pd.Series(predictions_each[0])
fig, ax = plt.subplots()
predictions_each_series.plot.bar()
fig.savefig('my_plot.png')
fig.show()


# In[11]:


import matplotlib.pyplot as plt
plt.plot(model.coef_[0])
plt.plot(model.coef_[1])
plt.plot(model.coef_[2])
plt.show()

