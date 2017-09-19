
# coding: utf-8

# #FEDS(Data analysis problem)
# ANSWERS are marked below in bold font. This is a code walk-through along with explanation

# In[7]:


import requests

URL = "https://www.w3schools.com/xml/simple.xml"

response = requests.get(URL)
with open('feed.xml', 'wb') as file:
    file.write(response.content)


# PROBLEM OVERVIEW-
# Data Analysis and Visualization ((Data: Order_Shipments.csv, Shipment_Deliveries.csv)
# As Amazon Fulfillment Execution, one of our core responsibilities is to make sure that shipments are delivered on time to the customers. For this purpose, we work very closely with our carriers to help them attain ad maintain a high level of on-time delivery performance. As part of this effort, you are tasked with measuring the on-time delivery performance of Amazon's contracted parcel carriers and creating actionable intelligence for operational leaders.
# 
# Director of Outbound Transportation has defined the following as key requirements for this metric:
# 1. Every carrier must have an on-time delivery performance score for each week of deliveries.
# 2. This score should reflect carriers' performance in terms of (in the order of importance - most important at the top):
# a. Existence of a customer complaint about the delivery experience (other than lateness)
# b. Delivery on or before promised delivery date and time
# c. Lateness (for those deliveries that are late by more than 1/2 hour)
# d. Earliness (for those deliveries that are early by more than 6 hours)
# 3. The score should be normalized to be between 0 and 100. Highest score should reflect the best
# performing carrier. and carriers must be ranked by the score for the score of the latest week (highest at the top).
# 4. The metric should show past 13 weeks of scores at the weekly level in tabular form as well as
# the Week-Over-Week change in score for the last week.
# 5. Overall delivery performance score (i.e. aggregate of all carriers) should be shown in a graph for
# the past 13 weeks.

# DATA OVERVIEW:-
# Historical delivery data is provided in the following files:
# Order_Shipments.csv: This file provides historical data on customer orders, shipments (an order can have multiple shipments) and their promised delivery date and time. Fields:
# Order_Datetime
# , Customer_Location_ID
# , Order_ID
# , Shipment_ID
# , Promised_Delivery_Datetime.
# ii. Shipment_Deliveries.csv: This file provides historical data on customer shipments, carriers, actual
# delivery date and time, and customer complaint counts. Each shipment is delivered by a single carrier.
# Fields:
# Shipment_ID
# , Carrier_ID
# , Actual_Delivery_Datetime
# , Shipment_Complaint_Count    

# PROBLEM TASK:-
# a) Please develop a computer application that reads the given files and produces the required output as shown above. You are welcome to use any framework/toolchain you would like as long as it is publicly available without a license. If you use R or Python you get extra credit since these are the languages we are currently utilizing.
# b) Given the provided data, develop a computer application that automatically identifies specific customer locations that have significantly difference overall performance than the rest of the customer landscape.

# STEP 1: DATASET ER MODEL

# In[1]:


#Import the libraries
from IPython.display import Image
import pandas as pd # dataframes
import numpy as np # algebra & calculus
import matplotlib.pyplot as plt # plotting


# In[2]:


Image(filename='Capture.JPG') 


# STEP2:  We will take a look at the dataset and merge both the dataset

# In[3]:


sdev = pd.read_csv('ShipmentDeliveries.csv')# Stores the shipment deliveries
sdev.head()


# In[4]:


oship = pd.read_csv('OrderShipments.csv')# Stores the order shipment
oship.head()


# In[5]:


dat=pd.merge(sdev,oship, on='Shipment_ID') # Merge dataset on Shipment_ID
dat.head()


# STEP 3: Here we will calculate
#     Delivery_Time=Promised_Delivery_Datetime-Actual_Delivery_Datetime
#     Delivery_beforeTime= If Delivery_Time is greater than 0
#     Lateness= If Delivery_Time is negative and less than 30 minutes(-0.5X60X60 seconds)
#     Earliness= If Delivery_Time is negative and less than 6 hours(-6X60X60 seconds)
#     Cust_comp_notlate= (Shipment_Complaint_Count-Lateness). When Lateness is 0 the value is just the Shipment_Complaint_Count.       
#     If lateness value is 1 and we have Shipment_complaint_count of 1 and more, I have assumed that count of 1 is due to             lateness. So I have substracted from Shipment_Complaint_Count-Lateness if its is 1 or more.

# In[6]:


dat.Carrier_ID.fillna('X', inplace=True) # Here carriers which are Null are replaced with X
#Convert object data type to datetime64
dat['Order_Datetime'] =  pd.to_datetime(dat['Order_Datetime'],yearfirst=True)
dat['Actual_Delivery_Datetime'] =  pd.to_datetime(dat['Actual_Delivery_Datetime'],yearfirst=True)
dat['Promised_Delivery_Datetime'] =  pd.to_datetime(dat['Promised_Delivery_Datetime'])
#Delivery time stores the before time delivery as postive value and after time as negative value 
dat['Delivery_Time'] = (dat['Promised_Delivery_Datetime']-dat['Actual_Delivery_Datetime']).astype('timedelta64[s]')
#Covert Delivery_beforeTime to 1(If delivered before time) and 0(If delivered after time)
dat['Delivery_beforeTime']=(dat['Delivery_Time']>=0)*1
dat['Lateness']=(dat['Delivery_Time']<-0.5*60*60)*1
dat['Earliness']=(dat['Delivery_Time']>6*60*60)*1
#Cust_comp_notlate stores a customer complaint count about the delivery experience (other than lateness)
dat['Cust_comp_notlate'] =(dat['Shipment_Complaint_Count']-dat['Lateness'])*((dat['Shipment_Complaint_Count']-dat['Lateness'])>0)
#Get the week number of each date.
#Note: From January-1-2016 to January-9-2016 is considered a week 1. Weeks are from Saturday to Sunday
dat['Actual_Delivery_Week'] =  pd.to_datetime(dat['Actual_Delivery_Datetime']).dt.week
print("Datatypes of each column-\n",dat.dtypes)
dat.head()


# In[7]:


#Maximum customer complaint value when package is not delivered late
print("Max and min range of customer complaint \n")
print(dat.Cust_comp_notlate.max())
#Minimum customer complaint value when package is not delivered late
print(dat.Cust_comp_notlate.min())
#Therefore there can be 3 X Total number of deliveries possible cases


# In[8]:


x=dat.groupby(['Actual_Delivery_Week','Carrier_ID']).agg({'Shipment_ID':np.size,'Lateness':np.sum, 'Earliness': np.sum,'Delivery_beforeTime': np.sum,'Cust_comp_notlate':np.sum})
#Here for week 1, Carrier_ID- GNU Logistics - We have a total of 25 Shipments. So probability of lateness is 3/25, earliness is 
# 1/25, delivery before time is 22/25, for each delivery there are 0 to 3 complaint types so cust_comp_notlate is 9/(25X3)
#Given:-
#a. Existence of a customer complaint about the delivery experience (other than lateness)- When value is more it is worse
#b. Delivery on or before promised delivery date and time- When value is more, it is good
#c. Lateness (for those deliveries that are late by more than 1/2 hour)- When value is more it is worse 
#d. Earliness (for those deliveries that are early by more than 6 hours)-When value is more it is worse. Too much early is not good.
x


# STEP 4: Here we will calculate the PERFORMANCE score

# COST FUNCTION:-
# PERFORMANCE=40*(1-Cust_comp_notlate)+30*(Delivery_beforeTime)+20*(1-Lateness)+10*(1-Earliness)
# Here Cust_comp_notlate,Delivery_beforeTime,Lateness,Earliness are normalized between 0 and 1 based on probability of occurence. 
# The multiplication factor of 40,30,20,10 are assigned based on importance. The exact values are chosen arbitarily as no background information was given.
# Below pLateness, pEarliness,pDelivery_beforeTime,pCust_comp_notlate are normalized value between 0 and 1. These are calculated by considering probability of  values-(Lateness, Earliness, Delivery_beforeTime) to total shipment(Shipment_ID in the next table) for a particular carrrier in a particular week. For pCust_comp_notlate, I have taken the ratio of Cust_comp_notlate and Shipment_ID X 3 since a customer complaint can range from 0 to 3.

# In[9]:


x['pLateness']=(x['Lateness']/x['Shipment_ID'])
x['pEarliness']=(x['Earliness']/x['Shipment_ID'])
x['pDelivery_beforeTime']=x['Delivery_beforeTime']/x['Shipment_ID']
x['pCust_comp_notlate']=(x['Cust_comp_notlate']/(x['Shipment_ID']*3))
x['PERFORMANCE']=(40*(1-x['pCust_comp_notlate'])+30*(x['pDelivery_beforeTime'])+20*(1-x['pLateness'])+10*(1-x['pEarliness']))
x.sort_values(['PERFORMANCE'], ascending=[1])
#df.sort(['A', 'B'], ascending=[1, 0])
x.iloc[:10]


# Here  I have shown the values grouped by Actual Delivery Week and shipments for each week denoted by Shipment_ID

# In[10]:


xsum=dat.groupby(['Actual_Delivery_Week']).agg({'Shipment_ID':np.size,'Lateness':np.sum, 'Earliness': np.sum,'Delivery_beforeTime': np.sum,'Cust_comp_notlate':np.sum})
xsum['pLateness']=(xsum['Lateness']/xsum['Shipment_ID'])
xsum['pEarliness']=(xsum['Earliness']/xsum['Shipment_ID'])
xsum['pDelivery_beforeTime']=xsum['Delivery_beforeTime']/xsum['Shipment_ID']
xsum['pCust_comp_notlate']=(xsum['Cust_comp_notlate']/(xsum['Shipment_ID']*3))
xsum['PERFORMANCE']=(40*(1-xsum['pCust_comp_notlate'])+30*(xsum['pDelivery_beforeTime'])+20*(1-xsum['pLateness'])+10*(1-xsum['pEarliness']))
xsum.sort_values(['PERFORMANCE'], ascending=[1])

xsum.head()


# # ANSWER a: PART 1

# Here the user is asked to enter the Week number. The week number is greater than 13 to show all the previous weeks. The week number can range from 14 to 52. For the year of 2016 Weeks start from 1 to 52. In the x-axis I have shown the week number for the year.(NOTE: It gives more information than just showing week number from 1 to 13). In the y-axis the values are shown from 85 to 100 as all the values are in this range.

# In[11]:


var = int(input("Please enter Week number: "))
print("you entered", var)
start=(var-14)
stop=(var-1)
plt.title('Carrier Delivery performance metric -Week 13')
plt.plot(xsum['PERFORMANCE'].iloc[start:stop])
plt.axis([start,stop,85,100])
plt.ylabel('some numbers')
plt.show()


# # ANSWER a: PART 2

# Step 6 : Here carrier-wise weekly number of parcels is shown

# In[12]:


x1=dat.groupby(['Carrier_ID','Actual_Delivery_Week']).agg({'Shipment_ID':np.size,'Lateness':np.sum, 'Earliness': np.sum,'Delivery_beforeTime': np.sum,'Cust_comp_notlate':np.sum})
x1['pLateness']=(x1['Lateness']/x1['Shipment_ID'])
x1['pEarliness']=(x1['Earliness']/x1['Shipment_ID'])
x1['pDelivery_beforeTime']=x1['Delivery_beforeTime']/x1['Shipment_ID']
x1['pCust_comp_notlate']=(x1['Cust_comp_notlate']/(x1['Shipment_ID']*3))
x1['PERFORMANCE']=(40*(1-x1['pCust_comp_notlate'])+30*(x1['pDelivery_beforeTime'])+20*(1-x1['pLateness'])+10*(1-x1['pEarliness']))
x1.sort_values(['PERFORMANCE'], ascending=[1])   
x2=x1['PERFORMANCE'].unstack()
x2 = x2[x2.columns[start:stop]]
x2['wow']=x2[stop]-x2[stop-1]
print(x2)


# # ANSWER b

# STEP 6: Here based on performance score the customer location is shown

# In[13]:


x3=dat.groupby(['Customer_Location_ID']).agg({'Shipment_ID':np.size,'Lateness':np.sum, 'Earliness': np.sum,'Delivery_beforeTime': np.sum,'Cust_comp_notlate':np.sum})
x3['pLateness']=(x3['Lateness']/x3['Shipment_ID'])
x3['pEarliness']=(x3['Earliness']/x3['Shipment_ID'])
x3['pDelivery_beforeTime']=x3['Delivery_beforeTime']/x3['Shipment_ID']
x3['pCust_comp_notlate']=(x3['Cust_comp_notlate']/(x3['Shipment_ID']*3))
x3['PERFORMANCE']=(40*(1-x3['pCust_comp_notlate'])+30*(x3['pDelivery_beforeTime'])+20*(1-x3['pLateness'])+10*(1-x3['pEarliness']))
print('Minimum performance score', x3['PERFORMANCE'].min())
print('Maximum performance score', x3['PERFORMANCE'].max())
print(x3)


# In[14]:


x3.to_csv('out.csv') #Save the output file


# In[ ]:




