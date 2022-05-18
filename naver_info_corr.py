#!/usr/bin/env python
# coding: utf-8

# In[65]:


from selenium import webdriver
from bs4 import BeautifulSoup as bs
from urllib.parse import quote
import requests
import time
import pandas as pd
import re
import csv
import ssl
import time
ssl._create_default_https_context = ssl._create_unverified_context


driver = webdriver.Chrome("/opt/homebrew/bin/chromedriver")
# driver = webdriver.Chrome("c:/pydata/chromedriver")
review_data=[]

kakaopage=[146656,61476,143722,110137,144508,107497,111490,121982,150296,122296]

for k in kakaopage:
    driver.get("https://movie.daum.net/moviedb/crew?movieId="+str(k))
    #driver.execute_script("window.scrollTo(0, 53)") 
    time.sleep(3)
    html = driver.page_source
    soup=bs(html,"html.parser")
    
    # 영화제목
    title=soup.find("span",class_='txt_tit').text
    
    # 주장르
    genre=soup.find_all("dl",class_="list_cont")[1]
    genres=genre.text.replace("\n","").split("/")[0]
    genre=genres.replace("장르","")
    
    # 주연배우
    mainactor_soup=soup.find_all("div", class_="item_crew")
    main_actor=mainactor_soup[1].find("strong", class_="tit_item")
    
    try:
        main_actor=main_actor.text.strip()
            
    except AttributeError as err:
         main_actor="주연배우미상"
             
    # 감독
    director_soup=soup.find("div",class_="item_crew")
    director=director_soup.find("a",class_="link_txt").text
#     print(director)
    
    # 배급사
    distributor_soup=soup.find_all("dl",class_="list_produce")[-2]
    distributors=distributor_soup.find("a", class_="link_txt").text
    
  
    review_data.append([title,distributors,genre,director,main_actor])

driver.close()
        
df=pd.DataFrame(review_data,columns=["title","distributor","genre","director","main_actor"])
display(df.head(10))
display(df.tail(10))

df.to_csv("./kakaopageranking_test.csv",header=True,index=False,encoding="utf8")
# df.to_excel("./kakaopagerank_test.xlsx",header=True,index=False)

# print("작업이 완료되었습니다.")


# In[33]:


get_ipython().system('ls')


# In[49]:


pwd!


# In[66]:


import pandas as pd


# In[67]:


naver_df=pd.read_csv("Main_actor_test.csv")
naver_df=naver_df.drop(columns=['Unnamed: 0'],axis=1)
daum_df=pd.read_csv("kakaopageranking_test.csv")


# In[100]:


naver_df=naver_df.head(10)
naver_df


# In[112]:


naver_df=naver_df.fillna("미상")
naver_df


# In[107]:


from nltk import FreqDist
import numpy as np

vocab=FreqDist(np.hstack(naver_df))
print(vocab)


# In[108]:


vocab_size=5
vocab=vocab.most_common(vocab_size)

word_to_index={word[0]:index+1 for index,word in enumerate(vocab)}
print(word_to_index)


# In[114]:


from sklearn.preprocessing import LabelEncoder


# In[125]:


# 라벨 인코더 생성
encoder = LabelEncoder()

# X_train데이터를 이용 피팅하고 라벨숫자로 변환한다
encoder.fit(naver_df["genre"])
X_train_encoded = encoder.transform(naver_df["genre"])
print(X_train_encoded)
# X_test데이터에만 존재하는 새로 출현한 데이터를 신규 클래스로 추가한다 (중요!!!)
for label in np.unique(naver_df["genre"]):
    if label not in encoder.classes_: # unseen label 데이터인 경우( )
        encoder.classes_ = np.append(encoder.classes_, label) # 미처리 시 ValueError발생
X_test_encoded = encoder.transform(naver_df["genre"])
X_test_encoded

val=pd.DataFrame()
val['genre_lab']=X_test_encoded


# In[126]:


# 라벨 인코더 생성
encoder = LabelEncoder()

# X_train데이터를 이용 피팅하고 라벨숫자로 변환한다
encoder.fit(naver_df["director"])
X_train_encoded = encoder.transform(naver_df["director"])
print(X_train_encoded)
# X_test데이터에만 존재하는 새로 출현한 데이터를 신규 클래스로 추가한다 (중요!!!)
for label in np.unique(naver_df["director"]):
    if label not in encoder.classes_: # unseen label 데이터인 경우( )
        encoder.classes_ = np.append(encoder.classes_, label) # 미처리 시 ValueError발생
X_test_encoded = encoder.transform(naver_df["director"])
X_test_encoded

# val=pd.DataFrame()
val['director_lab']=X_test_encoded


# In[130]:


# 라벨 인코더 생성
encoder = LabelEncoder()

# X_train데이터를 이용 피팅하고 라벨숫자로 변환한다
encoder.fit(naver_df["main_actor"])
X_train_encoded = encoder.transform(naver_df["main_actor"])
print(X_train_encoded)
# X_test데이터에만 존재하는 새로 출현한 데이터를 신규 클래스로 추가한다 (중요!!!)
for label in np.unique(naver_df["main_actor"]):
    if label not in encoder.classes_: # unseen label 데이터인 경우( )
        encoder.classes_ = np.append(encoder.classes_, label) # 미처리 시 ValueError발생
X_test_encoded = encoder.transform(naver_df["main_actor"])
X_test_encoded

# val=pd.DataFrame()
val['main_actor_lab']=X_test_encoded


# In[131]:


val


# In[132]:


corr=val.corr()
print(corr)

#1에 가까우면 상관이 있음, 0에 가까우면 상관이 없음, -면 상관관계가 반대임 


# In[133]:


# 라벨 인코더 생성
encoder = LabelEncoder()

# X_train데이터를 이용 피팅하고 라벨숫자로 변환한다
encoder.fit(daum_df["genre"])
X_train_encoded = encoder.transform(daum_df["genre"])
print(X_train_encoded)
# X_test데이터에만 존재하는 새로 출현한 데이터를 신규 클래스로 추가한다 (중요!!!)
for label in np.unique(daum_df["genre"]):
    if label not in encoder.classes_: # unseen label 데이터인 경우( )
        encoder.classes_ = np.append(encoder.classes_, label) # 미처리 시 ValueError발생
X_test_encoded = encoder.transform(daum_df["genre"])
X_test_encoded

yal=pd.DataFrame()
yal['genre_lab']=X_test_encoded


# In[137]:


# 라벨 인코더 생성
encoder = LabelEncoder()

# X_train데이터를 이용 피팅하고 라벨숫자로 변환한다
encoder.fit(daum_df["main_actor"])
X_train_encoded = encoder.transform(daum_df["main_actor"])
print(X_train_encoded)
# X_test데이터에만 존재하는 새로 출현한 데이터를 신규 클래스로 추가한다 (중요!!!)
for label in np.unique(daum_df["main_actor"]):
    if label not in encoder.classes_: # unseen label 데이터인 경우( )
        encoder.classes_ = np.append(encoder.classes_, label) # 미처리 시 ValueError발생
X_test_encoded = encoder.transform(daum_df["main_actor"])
X_test_encoded

# yal=pd.DataFrame()
yal['main_actor_lab']=X_test_encoded


# In[138]:


yal


# In[139]:


corr=yal.corr()
print(corr)


# In[87]:


naver_df.keys()


# In[88]:


naver_df.shape


# In[92]:


naver_df["distributor"].shape


# In[96]:


daum_df["distributor"].shape


# In[83]:


daum_df.shape


# In[191]:


X = val[["director_lab","main_actor_lab"]]
Y = val["genre_lab"]
print(Y)
x_train= X[:7].values
x_test=X[7:].values

y_train=Y[0:7].values
y_test=Y[7:].values

# x_train.shape,y_train.shape


# In[161]:


# 머신러닝
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings(action='ignore')


# In[192]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
Y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
acc_log


# In[193]:


val


# In[194]:


u1=np.array([[0,1]])
print(u1.shape)
logreg.predict(u1)


# In[ ]:




