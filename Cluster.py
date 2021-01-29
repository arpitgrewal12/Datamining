#ARPIT KAUR
#301367803

import pandas as pd
import numpy as np
import random
import operator
from collections import Counter
import copy

data=pd.read_csv('agaricus-lepiota.data',header=None)


data.columns=['class','cap-shape','cap-surface','cap-color','bruises?','odor','gill-attachment',
           'gill-spacing','gill-size','gill-color','stalk-shape','stalk-root',
           'stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring',
           'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type',
           'spore-print-color','population','habitat']

#FOR DATA IMPUTATION
rslt_df1 = data.loc[data['class'] =='?'] 
rslt_df2 = data.loc[data['cap-shape'] =='?'] 
rslt_df3 = data.loc[data['cap-surface'] =='?'] 
rslt_df4 = data.loc[data['cap-color'] =='?'] 
rslt_df5 = data.loc[data['bruises?'] =='?'] 
rslt_df6 = data.loc[data['odor'] =='?'] 
rslt_df7 = data.loc[data['gill-attachment'] =='?'] 
rslt_df8 = data.loc[data['gill-spacing'] =='?'] 
rslt_df9 = data.loc[data['gill-size'] =='?'] 
rslt_df10 = data.loc[data['gill-color'] =='?'] 
rslt_df11= data.loc[data['stalk-shape'] =='?'] 
rslt_df12 = data.loc[data['stalk-root'] =='?'] 
rslt_df13 = data.loc[data['stalk-surface-above-ring'] =='?'] 
rslt_df14 = data.loc[data['stalk-surface-below-ring'] =='?'] 
rslt_df15 = data.loc[data['stalk-color-above-ring'] =='?'] 
rslt_df16 = data.loc[data['stalk-color-below-ring'] =='?'] 
rslt_df17 = data.loc[data['veil-type'] =='?'] 
rslt_df18 = data.loc[data['veil-color'] =='?'] 
rslt_df19 = data.loc[data['ring-number'] =='?'] 
rslt_df20 = data.loc[data['ring-type'] =='?'] 
rslt_df21= data.loc[data['spore-print-color'] =='?'] 
rslt_df22 = data.loc[data['population'] =='?'] 
rslt_df23 = data.loc[data['habitat'] =='?'] 

#print(rslt_df12)

data['stalk-root'].replace({'?': None},inplace =True)
rslt_df12a = data.loc[data['stalk-root'] =='?'] 
#print(rslt_df12a)
#print(data.isnull().sum())

def impute(column1, column2):
    f = []
    for i in list(set(data[column1])):
        data_category = data[data[column1]== i]   
        data_category[column2].fillna(data_category[column2].mode()[0],inplace = True)        
        f.append(data_category)
        df = pd.concat(f)
    return df

data1 = impute('class', 'stalk-root')
#print('The number of missing values are:')
#print(data1.isnull().sum())
#Using data without the column label
data2=data1[data1.columns[1:23]]
#print(data2)

#Initialization of centroids
def initialize_centroids(X, n_clusters): 
    return np.array(random.sample(X, n_clusters))


#Calculating dissimilarities
def distances(X, centroids):
    dist = np.zeros(centroids.shape[0])
    #print(dist)
    for i in range(centroids.shape[0]):
        for j in range(centroids.shape[1]):
            if X[j] != centroids[i, j]:
                dist[i] += 1;
    return dist

#Assignment of observations to a cluster
def assignment(X, centroids):
    dist=distances(X,centroids)
    y=min(enumerate(dist), key=operator.itemgetter(1))[0]
    return y

#Iteration termination condition
def isequal(centroids,old_centroids):
    return set([tuple(a) for a in centroids])==set([tuple(b) for b in old_centroids])



#Computing new centroids
def update_centroids(X, a, centroid):
    centroids = copy.deepcopy(centroid)
    for i in range(centroids.shape[0]):
        points = np.array([X[j, :] for j in range(X.shape[0]) if a[j] == i])
        #print([i],'value of points is:')
        #print(points[i])
        for k in range(points.shape[1]):
            temp_points = [points[j, k] for j in range(points.shape[0])]
            #print(temp_points)
            c=Counter(temp_points)
            centroids[i, k] = max(c, key=c.get)[0]
    return centroids

#Kmodes algorithm
def kmodes(data, noOfClusters):
    #Initializing clusters
    X=list(data)
    old_centroids=initialize_centroids(X, noOfClusters)
    centroids=initialize_centroids(X, noOfClusters)
    iter=0
    #Assigning clusters
    df=np.array(data)
    d=np.zeros((len(data),noOfClusters))
    a=np.zeros((len(data),1))
    belongs_to = np.zeros((len(data), 1))
    while not isequal(centroids,old_centroids):
        old_centroids = centroids
        for i in range(len(data)):
            for j in range(noOfClusters):
                d[i][j]=distances(df[i,:],old_centroids)[j]
            a[i]=assignment(df[i,:],old_centroids)
            if belongs_to[i] != a[i]:
                belongs_to[i]= a[i]
        centroids=update_centroids(data,belongs_to,old_centroids)
        iter+=1
    print('Updated centroids:')  
    print(centroids)

    print ("Distances:")   
    print(d)
    print ("Assigning observations to clusters:")
    print(a)

    return centroids,belongs_to

data3=np.array(data2)
centroids,y = kmodes(data3, 5)
data['Assignment']=y

#Saving output to files
print("The data is saved in the Assignments.csv output file ")
gfg_csv_data = data.to_csv('Assignments.csv', index = False)
print('\nCSV String:\n', gfg_csv_data) 
