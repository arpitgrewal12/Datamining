

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys



#Reading the data
data=pd.read_csv('outliers-3.csv')


#Assigning column values to variables
x_values=data['X1']
y_values=data['X2']
print("The data shape is",data.shape)




k=7


# In[5]:


#Making empty lists to store data
reach_dist=[]
local_reach_dist=[]
local_outlier_factor=[]
k_neighbours=[]
k_neighbours_dist=[]
outlier=[]
threshold=2
#Initialising temporary list so that a matrix can be made for reachability distance
temporary_list=[]
outlier_x=[]
outlier_y=[]
nonoutlier_x=[]
nonoutlier_y=[]


# In[6]:


for i in range(len(x_values)):
    for j in range(len(y_values)):
        temporary_list.append(0)
    local_reach_dist.append(0)
    local_outlier_factor.append(-1)
    k_neighbours.append(0)
    k_neighbours_dist.append(0)
    #Reachability distance matrix because 
    #reachability distance of A from B is not equal to the reachability distance of B from A
    reach_dist.append(temporary_list)
    temporary_list=[]


# In[7]:


#This function calculates the neighbours in the k-distance neighbourhood of any given point
def find_nearest_k(i,x_values,k):
    k_distance=0
    d_values=dict()
    neighbours=[]
    neighbour_dist=[]
    
    for j in range (len(x_values)):
        if i==j:
            continue
        else:
            dist=abs(x_values.iloc[i]-x_values.iloc[j])+abs(y_values.iloc[i]-y_values.iloc[j])
            d_values[j]=dist
            
    j=1
    sort=sorted(d_values.items(),key=lambda item:item[1],reverse=False)
    d_values={u:v for u,v in sort}
    
    for u,v in d_values.items():
        if j>k and v!=k_distance:
            break
            
        if j==k:
            k_distance=v
            
        neighbour_dist.append(v)
        neighbours.append(u)
        j=j+1
        
    return neighbours,neighbour_dist,k_distance
            


# In[8]:


def update_reach_dist(i,k_distance):
    var=0
    reach_dist_sum=0
    
    for j in k_neighbours[i]:
        reach_dist[j][i]=max(k_distance,k_neighbours_dist[i][var])
        reach_dist_sum=reach_dist_sum+reach_dist[j][i]
        var=var+1    
    return reach_dist_sum


# In[9]:


#Calculates the local reachability distance of every point and stored it in a list to avoid re-calculating it multiple times
def calc_local_rd(i,reach_dist_sum):
    r=reach_dist_sum/len(k_neighbours[i])
    local_reach_dist[i]=1/(r)


# In[10]:


def calc_lof_score(i):
    lrdsum=0
    for j in k_neighbours[i]:
        lrdsum=lrdsum+local_reach_dist[j]
    local_outlier_factor[i]=(lrdsum/local_reach_dist[i])/len(k_neighbours[i])


# In[11]:


for i in range (len(x_values)):
    if k_neighbours[i]==0:
        k_neighbours[i],k_neighbours_dist[i],k_distance=find_nearest_k(i,x_values,k)
    reach_dist_sum=update_reach_dist(i,k_distance)
    calc_local_rd(i,reach_dist_sum)
    
for i in range(len(x_values)):
    calc_lof_score(i)


# In[12]:


#Plotting graph
for i in range(len(local_outlier_factor)):
    if local_outlier_factor[i]>threshold:
        outlier.append(i)
        outlier_x.append(x_values.iloc[i])
        outlier_y.append(y_values.iloc[i])
    else:
        nonoutlier_x.append(x_values.iloc[i])
        nonoutlier_y.append(y_values.iloc[i])


# In[13]:


plt.scatter(nonoutlier_x,nonoutlier_y,c='blue',label='Non-Outliers')
plt.scatter(outlier_x,outlier_y,c='red',label='Outliers')
plt.legend()
plt.title('Plot showing outliers with k=7 and threshold=2')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




