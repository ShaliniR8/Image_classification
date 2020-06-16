#!/usr/bin/env python
# coding: utf-8

# **LOAD DATASET**

# In[264]:


import pandas as pd

train = pd.read_csv(r'C:\Users\KIIT\Downloads\Train.csv')
test = pd.read_csv(r'C:\Users\KIIT\Downloads\Test.csv')


# In[266]:


train_indices = train.shape[0]
test_indices = test.shape[0]
print("train_indices, test_indices = ", train_indices, test_indices )
total = pd.concat([train, test], axis=0)
print("total.shape = ",total.shape)


# **PREPROCESSING** (removing special characters)

# In[221]:


import re

def preprocessing(text):    
    text = re.sub('<[^>]*>','',text.lower())
    text = re.sub('[\W+^]',' ',text) 
    return text


# In[222]:


before = test['text'][55]
train['text'] = train['text'].apply(preprocessing)
test['text'] = test['text'].apply(preprocessing)


# In[223]:


print("Before : ", before[:300])
print("\nAfter : ",test['text'][55][:300])


# **TOKENIZE AND REMOVE STOPWORDS**

# In[224]:


import nltk
nltk.download('punkt')


# In[225]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop = set(stopwords.words('english'))

def my_swremove(text):
    text=word_tokenize(text)
    new_text =[ words for words in text if words not in stop]
    return new_text


# **STEMMING**

# In[226]:


from nltk.stem import PorterStemmer

port =  PorterStemmer()
def my_stemmer(text):
    
    stemmed = [ port.stem(words) for words in text]
    return stemmed


# **LEMMATIZER**

# In[227]:


from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()
def my_lemmatizer(text):
    lem_text = [ lem.lemmatize(words) for words in text ]
    return ' '.join(lem_text)


# **TF_IDF VECTOR**

# In[245]:


from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize(text):
    tfid = TfidfVectorizer( smooth_idf = True, use_idf = True, preprocessor = None)
    return tfid.fit_transform(text)


# **TESTING WITH AN EXAMPLE FIRST**

# In[255]:


s1 = "This is a sample sentence <br go to next line\>  to test all my functions :). I was running "
s2 = "This is my second sample sentence. I wanted to sing and now i am singing."

df = pd.DataFrame([s1,s2], columns = ['sentences'])

df['sentences'] = df['sentences'].apply(preprocessing)
df['sentences'] = df['sentences'].apply(my_swremove)
df['sentences'] = df['sentences'].apply(my_stemmer)
df['sentences'] = df['sentences'].apply(my_lemmatizer)
print(df['sentences'])

tfid = vectorize(df['sentences'])
print(tfid)


# **APPLYING ALL FUNCTIONS TO DATASET**

# In[267]:


total['text'] = total['text'].apply(my_swremove)
total['text'] = total['text'].apply(my_stemmer)
total['text'] = total['text'].apply(my_lemmatizer)


# In[268]:


X = vectorize(total['text'])
y = total['label']


# In[269]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size= 1/8)


# In[270]:


import pickle
from sklearn.linear_model import LogisticRegressionCV

lr = LogisticRegressionCV(
    cv = 4,
    scoring = 'accuracy',
    verbose = 3,
    max_iter = 300,
    n_jobs = -1
)

lr.fit(X_train, y_train)


# In[271]:


X_train


# In[272]:


saved_model = open( 'saved_model.sav', 'wb')
pickle.dump( lr, saved_model )
saved_model.close()


# In[273]:


file = 'saved_model.sav'
model = pickle.load(open( file, 'rb' ))


# In[274]:


model.score( X_test, y_test )


# In[ ]:




