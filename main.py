#!/usr/bin/env python
# coding: utf-8

# In[25]:


#Python  libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns


# In[45]:


#load dataset
data = pd.read_excel('dataset.xlsx')
data.head()


# In[46]:


#data exploration
df = data.copy()
df.shape


# In[47]:


df.dtypes


# In[48]:


df.dtypes.value_counts()


# In[49]:


df.dtypes.value_counts().plot.pie()


# In[50]:


df.isna()


# In[51]:


sns.heatmap(df.isna())


# In[33]:


df.isna().sum()


# In[52]:


(df.isna().sum()/df.shape[0]).sort_values(ascending=False)


# In[53]:


df[df.columns[df.isna().sum()/df.shape[0]<0.9]]


# In[54]:


sns.heatmap(df.isna(), cbar=False)


# In[55]:


df = df.drop('Patient ID', axis=1)
df.head()


# In[58]:


df['SARS-Cov-2 exam result'].value_counts()


# In[62]:


for col in df.select_dtypes('float'):
    plt.figure()
    sns.distplot(df[col])


# In[63]:


sns.distplot(df['Patient age quantile'])


# In[68]:


for col in df.select_dtypes('object'):
    print(f'{col :-<40} {df[col].unique()}')


# In[69]:


for col in df.select_dtypes('object'):
    plt.figure()
    df[col].value_counts().plot.pie()


# In[71]:


positive_df = df[df['SARS-Cov-2 exam result'] == 'positive']


# In[72]:


negative_df = df[df['SARS-Cov-2 exam result'] == 'negative']


# In[73]:


missing_rate = df.isna().sum()/df.shape[0]


# In[74]:


blood_columns = df.columns[(missing_rate < 0.9) & (missing_rate > 0.88)]


# In[75]:


viral_columns = df.columns[(missing_rate < 0.88) & (missing_rate > 0.75)]


# In[80]:


for col in blood_columns:
    plt.figure()
    sns.distplot(positive_df[col], label='positive')
    sns.distplot(negative_df[col], label='negative')
    plt.legend()


# In[81]:


sns.countplot(x='Patient age quantile', hue='SARS-Cov-2 exam result', data=df)


# In[82]:


pd.crosstab(df['SARS-Cov-2 exam result'], df['Influenza A'])


# In[90]:


for col in viral_columns:
    plt.figure()
    sns.heatmap(pd.crosstab(df['SARS-Cov-2 exam result'], df[col]), annot=True, fmt='d')


# In[92]:


sns.heatmap(df[blood_columns].corr())


# In[93]:


sns.clustermap(df[blood_columns].corr())


# In[95]:


for col in blood_columns:
    plt.figure()
    sns.lmplot(x='Patient age quantile', y=col, hue='SARS-Cov-2 exam result', data=df)


# In[98]:


df.corr()['Patient age quantile'].sort_values()


# In[99]:


sns.pairplot(df[blood_columns])


# In[150]:


# preprocessing
df1 = data.copy()


# In[151]:


missing_rate = df1.isna().sum()/df1.shape[0]


# In[152]:


blood_columns = list(df1.columns[(missing_rate<0.9) & (missing_rate>0.88)])


# In[153]:


viral_columns = list(df1.columns[(missing_rate<0.88) & (missing_rate>0.75)])


# In[154]:


key_columns = ['Patient age quantile', 'SARS-Cov-2 exam result']


# In[155]:


df1= df1[key_columns + blood_columns + viral_columns]


# In[156]:


df1.head()


# In[183]:


from sklearn.model_selection import train_test_split


# In[184]:


trainset, testset = train_test_split(df1, test_size=0.2, random_state=0)


# In[185]:


trainset['SARS-Cov-2 exam result'].value_counts()


# In[186]:


testset['SARS-Cov-2 exam result'].value_counts()


# In[161]:


code = {'positive':1,'negative':0,'detected':1,'not_detected':0}


# In[162]:


for col in df1.select_dtypes('object'):
    df1[col] = df1[col].map(code)


# In[187]:


df1.dtypes.value_counts()


# In[188]:


def encodage(df):
    code = {'positive':1,'negative':0,'detected':1,'not_detected':0}
    for col in df.select_dtypes('object'):
        df[col] = df[col].map(code)
    return df


# In[189]:


def imputation(df):
    return df.dropna(axis=0)


# In[190]:


def preprocessing(df):
    df = encodage(df)
    df = imputation(df)
    
    X = df.drop('SARS-Cov-2 exam result', axis=1)
    y = df['SARS-Cov-2 exam result']
    
    print(y.value_counts())
    
    print(X, y)
    return X,y


# In[191]:


X_train, y_train = preprocessing(trainset)


# In[192]:


X_test, y_test = preprocessing(testset)


# In[271]:


# modèles
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# In[272]:


d_tree = DecisionTreeClassifier(random_state=0)


# In[273]:


# évaluation
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve


# In[274]:


def evaluation(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# In[275]:


evaluation(d_tree)


# In[276]:


list_models = []


# In[277]:


preprocessor = make_pipeline(PolynomialFeatures(2, include_bias = False), SelectKBest(f_classif, k=10))


# In[278]:


RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))


# In[279]:


AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))


# In[280]:


SVM = make_pipeline(preprocessor, StandardScaler(), SVC(random_state=0))


# In[281]:


KNN = make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())


# In[282]:


dict_models = {'RandomForest': RandomForest, 'AdaBoost': AdaBoost, 'SVM': SVM, 'KNN': KNN}


# In[284]:


for key, value in dict_models.items():
    print(key)
    evaluation(value)


# In[ ]:




