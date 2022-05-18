#!/usr/bin/env python
# coding: utf-8

# In[54]:


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

# df.to_csv("./kakaopagerank_test.csv",header=True,index=False,encoding="utf8")
# df.to_excel("./kakaopagerank_test.xlsx",header=True,index=False)

# print("작업이 완료되었습니다.")


# In[33]:


get_ipython().system('ls')


# In[49]:


pwd!


# In[50]:


import pandas as pd


# In[51]:


naver_df=pd.read_csv("expected_test")
daum_df=pd.read_csv("kakaopagerank_test")


# In[ ]:




