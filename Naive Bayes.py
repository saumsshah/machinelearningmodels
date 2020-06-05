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


# test train data. Run this code for bi-grams

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(headlines['Headline'], headlines['label'] , random_state = 42,test_size = 0.2)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range =(2, 2))
X_train_cv = cv.fit_transform(X_train,y=None)
X_test_cv = cv.transform(X_test)

from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(ngram_range =(2, 2))
#tf = tf1.fit(X_train_cv)
X_train_tv = tv.fit_transform(X_train)
X_test_tv = tv.transform(X_test)


from sklearn.feature_extraction.text import TfidfTransformer
tf = TfidfTransformer().fit(X_train_tv)
X_train_tf = tf.fit_transform(X_train_tv)
X_test_tf = tf.fit_transform(X_test_tv)
X_train_tf.shape


# In[ ]:


# test train data. Run this code for uni-grams


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(headlines['Headline'], headlines['label'] , random_state = 42,test_size = 0.2)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words= "english")
X_train_cv = cv.fit_transform(X_train,y=None)
X_test_cv = cv.transform(X_test)

from sklearn.feature_extraction.text import TfidfTransformer
tf = TfidfTransformer().fit(X_train_cv)
X_train_tf = tf.fit_transform(X_train_cv)
X_test_tf = tf.fit_transform(X_test_tv)
X_train_tf.shape


# In[5]:


#multinomialNB

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tf, y_train)
predictions = naive_bayes.predict(X_test_tf)


# In[ ]:


#BernoulliNB

from sklearn.naive_bayes import BernoulliNB
naive_bayes = BernoulliNB()
naive_bayes.fit(X_train_tf, y_train)
predictions = naive_bayes.predict(X_test_tf)


# In[ ]:


#GaussianNB

from sklearn.naive_bayes import GaussianNB
naive_bayes = BernoulliNB()
naive_bayes.fit(X_train_tf, y_train)
predictions = naive_bayes.predict(X_test_tf)


# In[6]:


#Accuracy

from sklearn.metrics import accuracy_score, precision_score, recall_score
print('Accuracy score: ', accuracy_score(y_test, predictions))


# In[7]:


# Converting numbers to values

testing_predictions = []
for i in range(len(X_test)):
    if predictions[i] == 2:
        testing_predictions.append('Low')
    elif predictions[i] == 1:
        testing_predictions.append('Medium')
    else:
        testing_predictions.append('High')
check_df = pd.DataFrame({'actual_label': list(y_test), 'prediction': testing_predictions, 'Headline':list(X_test)})
check_df.replace(to_replace=0, value='High', inplace=True)
check_df.replace(to_replace=1, value='Medium', inplace=True)
check_df.replace(to_replace=2, value='Low', inplace=True)


# In[9]:


#Confusion matrix and individual accuracy

y_test.replace(to_replace=0, value='High', inplace=True)
y_test.replace(to_replace=1, value='Medium', inplace=True)
y_test.replace(to_replace=2, value='Low', inplace=True)

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

y_true = y_test.to_numpy()
y_pred = testing_predictions
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


# In[11]:


#Single value prediction for uni-gram

data = ["MI: The Daily Dose Latin America: Cielo's Q1 profit plunges; Paraguay GDP contraction seen"]
testHeadline = pd.DataFrame(data,columns = ['Headline'])
X_static_test = testHeadline['Headline']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words= "english")
X_train_cv = cv.fit_transform(X_train,y=None)
X_statictest_cv = cv.transform(X_static_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
model = LogisticRegression()
model.fit(X_train_tf,y_train)
predictions = model.predict(X_statictest_cv)
individual_predictions = model.predict_proba(X_statictest_cv)

testing_predictions = []
for i in range(len(X_static_test)):
    if predictions[i] == 2:
        testing_predictions.append('Low')
    elif predictions[i] == 1:
        testing_predictions.append('Medium')
    else:
        testing_predictions.append('High')

print (testing_predictions)
print (individual_predictions)


# In[12]:


#Single value prediction for bi-gram

data = ["MI: The Daily Dose Latin America: Cielo's Q1 profit plunges; Paraguay GDP contraction seen"]
testHeadline = pd.DataFrame(data,columns = ['Headline'])
X_static_test = testHeadline['Headline']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range =(2, 2))
X_train_cv = cv.fit_transform(X_train,y=None)
X_statictest_cv = cv.transform(X_static_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
model = LogisticRegression()
model.fit(X_train_tf,y_train)
predictions = model.predict(X_statictest_cv)
individual_predictions = model.predict_proba(X_statictest_cv)

testing_predictions = []
for i in range(len(X_static_test)):
    if predictions[i] == 2:
        testing_predictions.append('Low')
    elif predictions[i] == 1:
        testing_predictions.append('Medium')
    else:
        testing_predictions.append('High')

print (testing_predictions)
print (individual_predictions)


# In[ ]:




