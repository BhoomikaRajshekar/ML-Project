#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # The Email spam detection problem is a set of Email meesages need to be tagged as "spam" and "ham"

# ## Importing Lib

# In[61]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
np.set_printoptions(threshold=np.inf)


# ## Import Data 

# In[62]:


data=pd.read_csv("G:/html form/spam.csv",encoding=('latin-1'))


# # Data Cleaning

# In[63]:


df_sms=data
df_sms.head()


# In[64]:


#drop unwanted columns
df_sms=df_sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df_sms = df_sms.rename(columns={"v1":"label", "v2":"sms"})
df_sms.head()


# ## Check for missing values

# In[65]:


df_sms.isnull().sum()


# In[66]:


df_sms.shape


# In[67]:


df_sms.info()


# In[68]:


import seaborn as sns
sns.countplot(x="label", data=df_sms)


# In[69]:


df_sms.describe()


# In[70]:


df_sms['length'] = df_sms['sms'].apply(len)
print(df_sms.head())
df_sms.hist(column='length', by='label', bins=5,figsize=(10,5))


# changing label values to 0 and 1

# In[71]:


df_sms.loc[:,'label'] = df_sms.label.map({'ham':0, 'spam':1})
print(df_sms.shape)
df_sms.head()


# splitting data into train and test (80-20)

# In[72]:



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_sms['sms'], 
                                                    df_sms['label'],test_size=0.20, 
                                                    random_state=1)

#train_test_split(x, y, test_size)
X_train.head()


# processing data using CountVectorizer and TfidfVectorizer which will be the input for algorithm

# In[73]:


count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)
# count and fit
# Transform testing data and return the matrix. 
testing_data = count_vector.transform(X_test)


# In[74]:


a=training_data.toarray()
a[[2]]


# In[75]:


count_vector.inverse_transform(a[[2]])


# In[76]:


count_vector=TfidfVectorizer(stop_words='english')
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)


# In[77]:


#count_vector.vocabulary_


# building MultinomialNB model

# In[78]:


naive_bayes = MultinomialNB()
naive_bayes.fit(training_data,y_train)
predictions = naive_bayes.predict(testing_data)


# In[79]:


actual=np.array(y_test)
actual


# In[80]:


predictions


# In[81]:


y_test.shape


# In[82]:


c=0
for i in range(len(predictions)):
  if predictions[i]==actual[i]:
    c=c+1
c


# In[83]:


1093/1115


# In[84]:


print('Accuracy score: {}'.format(accuracy_score(y_test, predictions)))
print('Precision score: {}'.format(precision_score(y_test, predictions)))
print('Recall score: {}'.format(recall_score(y_test, predictions)))
print('F1 score: {}'.format(f1_score(y_test, predictions)))


# In[85]:


from sklearn import metrics
print(metrics.classification_report(y_test, predictions))


# In[86]:


from sklearn.metrics import confusion_matrix
metrics.confusion_matrix(y_test, predictions)


# In[87]:


#consider entire records for training in order to get good prediction 


# In[88]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_sms['sms'], 
                                                    df_sms['label'],test_size=0.01, 
                                                    random_state=1)

#train_test_split(x, y, test_size)

count_vector = CountVectorizer()
# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)
# count and fit
# Transform testing data and return the matrix. 
testing_data = count_vector.transform(X_test)
count_vector=TfidfVectorizer(stop_words='english')
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data,y_train)
predictions = naive_bayes.predict(testing_data)


# In[91]:



import imaplib
import email
from bs4 import BeautifulSoup
import html2text

host = 'imap.gmail.com'
username = 'tstranger725@gmail.com'
password = 'cxqopbbifmdtyynn'

h = html2text.HTML2Text()
h.ignore_links = True

def get_inbox():
    mail = imaplib.IMAP4_SSL(host)
    mail.login(username, password)
    mail.select("inbox")
    _, search_data = mail.search(None, 'UNSEEN')
    my_message = []
    for num in search_data[0].split():
        email_data = {}
        _, data = mail.fetch(num, '(RFC822)')
        # print(data[0])
        _, b = data[0]
        email_message = email.message_from_bytes(b)
        for header in ['subject', 'to', 'from', 'date']:
            print("{}: {}".format(header, email_message[header]))
            email_data[header] = email_message[header]
        for part in email_message.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True)
                email_data['body'] = body.decode()
            elif part.get_content_type() == "text/html":
                html_body = part.get_payload(decode=True)

                soup = BeautifulSoup(html_body, features="html.parser")

                # kill all script and style elements
                for script in soup(["script", "style"]):
                    script.extract()    # rip it out

                # get text
                text = soup.get_text()

                email_data['html_body'] = h.handle(str(text))
                print( email_data['html_body'])
                

        my_message.append(email_data)
    return my_message
    

if __name__ == "__main__":
    my_inbox = get_inbox()
    print(my_inbox)


# In[92]:


my_inbox


# In[94]:


df = pd.DataFrame(my_inbox)
a=df.html_body.values[0]
df


# In[95]:


a=pd.Series(a)
a


# In[103]:


#a=pd.Series("free reward and a free mobile update woth $10")


# In[107]:


#a=pd.Series("good morning")


# In[96]:


testing_data_new = count_vector.transform(a)
pred=naive_bayes.predict(testing_data_new)
if pred[0]==1:
    print("spam")
else:
    print("ham")


# In[97]:


pred_prob=naive_bayes.predict_proba(testing_data_new)
# 0 , 1
# Ham , Spam
print(pred_prob)


# In[ ]:





# In[98]:


new_df=pd.DataFrame(pred_prob)
new_df


# In[99]:


new_df[0][0]


# In[104]:


if new_df[0][0]>=0.5:
    print("move to inbox")
elif (new_df[0][0]>0.5 or new_df[0][0]< 0.7) :
    print("spam send to indbox with warning")
elif (new_df[0][0]>=0.7 or new_df[0][0]<=0.9):
    print("spam send to spam")
else:
    print("delete spam email")


# In[52]:


# 50% - spam send to inbox
# 50%+ - 70%  spam send to indbox with warning
# 70% - 90% spam send to spam 
# 90% delete


# In[54]:


pred


# In[55]:


pred[0]


# In[ ]:




