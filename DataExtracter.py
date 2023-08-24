# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 17:36:39 2021

@author: KUMAR YASHASWI
"""

from bs4 import BeautifulSoup 
import requests
from requests import contentprint
from datetime import date

response = requests.get('http://google.com')

def main():  
    data_url = "https://finance.yahoo.com/quote/SPY/options"
    data_html = requests.get(data_url).content
    content = BeautifulSoup(data_html,"html.parser") 
    options_tables = content.find_all("table") 
    options_tables = [] 
    tables = content.find_all("table") 
    for i in range(0, len(content.find_all("table"))):   
        options_tables.append(tables[i])
    calls = options_tables[0].find_all("tr")[1:] # first row is header
    itm_calls = []
    otm_calls = []
    for call_option in calls:    
        if "in-the-money" in str(call_option):  
            itm_calls.append(call_option) 
        else:    
            otm_calls.append(call_option)
            
    itm_call = itm_calls[-1]
    #otm_call = otm_calls[0]
    itm_call_data = [] 
    for td in BeautifulSoup(str(itm_call), "html.parser").find_all("td"):   
        itm_call_data.append(td.text)
    itm_call_info = {'contract': itm_call_data[0], 'strike': itm_call_data[2], 'last': itm_call_data[3],  'bid': itm_call_data[4], 'ask': itm_call_data[5], 'volume': itm_call_data[8], 'iv': itm_call_data[10]}

{'contract': itm_call_data[0], 'strike': itm_call_data[2], 'last': itm_call_data[3],  'bid': itm_call_data[4], 'ask': itm_call_data[5], 'volume': itm_call_data[8], 'iv': itm_call_data[10], 
 'strike1': itm_call_data[1], 'last1': itm_call_data[6],  'bid1': itm_call_data[7], 'ask1': itm_call_data[9]}
'strike': itm_call_data[2], 'last': itm_call_data[3],  'bid': itm_call_data[4], 'ask': itm_call_data[5]

def main():  
    data_url = "https://finance.yahoo.com/quote/SPY/options"
    data_html = requests.get(data_url).content
    content = BeautifulSoup(data_html,"html.parser") 
    options_tables = content.find_all("table") 
    options_tables = [] 
    tables = content.find_all("table") 
    for i in range(0, len(content.find_all("table"))):   
        options_tables.append(tables[i])
    calls = options_tables[0].find_all("tr")[1:] # first row is header
    itm_calls = []
    otm_calls = []
    for call_option in calls:    
        if "in-the-money" in str(call_option):  
            itm_calls.append(call_option) 
        else:    
            otm_calls.append(call_option)
            
    #itm_call = itm_calls[-1]
    otm_call = otm_calls[0]
    otm_call_data = []
    for td in BeautifulSoup(str(otm_call), "html.parser").find_all("td"):   
        otm_call_data.append(td.text)
    otm_call_info = {'contract': otm_call_data[0], 'strike': otm_call_data[2], 'last': otm_call_data[3],  'bid': otm_call_data[4], 'ask': otm_call_data[5], 'volume': otm_call_data[8], 'iv': otm_call_data[10]}

puts = options_tables[1].find_all("tr")[1:] 
itm_puts = []  
otm_puts = []

for put_option in puts:    
    if "in-the-money" in str(put_option):      
        itm_puts.append(put_option)    
    else: 
        otm_puts.append(put_option)



otm_call
otm_call_data = []
for td in BeautifulSoup(str(otm_call), “html.parser”).find_all(“td”):  otm_call_data.append(td.text)

print(itm_call_info)

otm_call=itm_call
otm_call_data = []
for td in BeautifulSoup(str(otm_call), "html.parser").find_all("td"):  
    otm_call_data.append(td.text)
    
otm_call_info = {'contract': otm_call_data[0], 'strike': otm_call_data[2], 'last': otm_call_data[3],  'bid': otm_call_data[4], 'ask': otm_call_data[5], 'volume': otm_call_data[8], 'iv': otm_call_data[10]}

    
if __name__ == "__main__":  
    main()
    
data_url = "https://finance.yahoo.com/quote/SPY/options"
data_html = requests.get(data_url).content
print(data_html)

content = BeautifulSoup(data_html,"html.parser") 
 # print(content)

options_tables = content.find_all("table") 
print(options_tables)

options_tables = [] 
tables = content.find_all("table") 
for i in range(0, len(content.find_all("table"))):   
    options_tables.append(tables[i])
    
print(options_tables)
date.today()-10
datestamp="30032021"
expiration = datetime.datetime.fromtimestamp(int(datestamp)).strftime("%Y-%m-%d")

calls = options_tables[0].find_all("tr")[1:] # first row is header
itm_calls = []
otm_calls = []

for call_option in calls:    
    if "in-the-money" in str(call_option):  
        itm_calls.append(call_option) 
    else:    
        otm_calls.append(call_option)
        
itm_call = itm_calls[-1]
otm_call = otm_calls[0]

print(str(itm_call) + " \n\n " + str(otm_call))

itm_call_data = [] 
for td in BeautifulSoup(str(itm_call), "html.parser").find_all("td"):   
    itm_call_data.append(td.text)


print(itm_call_data)
itm_call_info = {'contract': itm_call_data[0], 'strike': itm_call_data[2], 'last': itm_call_data[3],  'bid': itm_call_data[4], 'ask': itm_call_data[5], 'volume': itm_call_data[8], 'iv': itm_call_data[10]}

print(itm_call_info)

otm_call=itm_call
otm_call_data = []
for td in BeautifulSoup(str(otm_call), "html.parser").find_all("td"):  
    otm_call_data.append(td.text)
    
otm_call_info = {'contract': otm_call_data[0], 'strike': otm_call_data[2], 'last': otm_call_data[3],  'bid': otm_call_data[4], 'ask': otm_call_data[5], 'volume': otm_call_data[8], 'iv': otm_call_data[10]}

print(itm_call_info)

import datetime, time

def get_datestamp():  
    options_url = "https://finance.yahoo.com/quote/SPY/options?date="  
    today = int(time.time())  
    print(today)  
    date = datetime.datetime.fromtimestamp(today)  
    yy = date.year  
    mm = date.month  
    dd = date.day
    
    dd -= 9
    
    options_day = datetime.date(yy, mm, dd)
    datestamp = int(time.mktime(options_day.timetuple())) 
    print(datestamp) 
    #print(datetime.datetime.fromtimestamp(options_stamp))
    
    if tables != []:   
        print(datestamp)   
        return str(datestamp) 
    else:   
        # print(“Bad datestamp!”)   
        dd += 1   
        options_day = datetime.date(yy, mm, dd)   
        datestamp = int(time.mktime(options_day.timetuple()))  
        return str(-1)

vet timestamp, then return if valid for i in range(0, 7):   
    test_req = requests.get(options_url + str(datestamp)).content   
    content = BeautifulSoup(test_req, “html.parser”)   # print(content)   tables = content.find_all(“table”)
    
if tables != []:   
    print(datestamp)   
    return str(datestamp) 
else:   
    # print(“Bad datestamp!”)   
    dd += 1   
    options_day = datetime.date(yy, mm, dd)   
    datestamp = int(time.mktime(options_day.timetuple()))  
    return str(-1)
    
datestamp = get_datestamp()
data_url = "https://finance.yahoo.com/quote/SPY/options?date=" + datestamp

options_list = {'calls': {'itm': itm_call_info, 'otm': otm_call_info}, 'date': datetime.date.fromtimestamp(time.time()).strftime("%Y-%m-%d")}
return options_list

  today = int(time.time()) 
  date = datetime.datetime.fromtimestamp(today) 
  yy = date.year 
  mm = date.month 
  dd = date.day
  
KCyG-a2ybzjiVDGiBXAy


from wallstreet import Stock, Call, Put
g = Call('GOOG', d=16, m=4, y=2021, strike=945)
g.price
g.underlying.price
g.date
g.strikes
g=g.set_strike(945)
g.expiration
g.volume
import quandl
