#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
from pprint import pprint


Interviewee=pd.read_csv("interviewee.csv")
Interviewee_list = np.array(Interviewee)
bankstest=pd.read_csv('banks-test.csv')
banks=pd.read_csv('banks.csv')
banks_list=np.array(banks)
bankstest_list=np.array(bankstest)

def entropy(data):
    data_list = np.array(data)
    labelcolumn = data_list[:, -1]
    elements, counts = np.unique(labelcolumn, return_counts=True)
    counts_sum=counts.sum()
    prob = counts / counts_sum
    entropy = sum(prob * -np.log2(prob))
    return entropy

#How much the question reduces the impurity is given by information gain
def info_gain(left, right, data):
    current=entropy(data)
    entropy_left=entropy(left)
    entropy_right=entropy(right)
    total_length=len(left) + len(right)
    probability_left = float(len(left) / total_length)
    probability_right=1-probability_left
    entropy_split=probability_left*entropy_left+probability_right* entropy_right
    information_gain=current-entropy_split
    return information_gain

attribute_names=list(banks)

#Defining a class
class Question:
    #Defining a constructor within the class
    def __init__(self, column, value, data):
        self.column = column
        self.value = value
        self.data= data
        
    def match(self, data):
        val = data[self.column]
        if (val == self.value):
            return val
    
    def __repr__(self):
        #a=attribute_names(self.data)
        # print(len(a))
        #b=self.column
        #print(range(b))
        condition = "=="
        return "Is %s %s %s?" % (attribute_names[self.column], condition, str(self.value))


def partition(data, question):
    data_list = np.array(data)
    true_values=[]
    false_values = []
    
    for i in data_list:
        if question.match(i):
            true_values.append(i)
        else:
            false_values.append(i)
    return true_values, false_values
#true_rows, false_rows = partition(bankstest, Question(0, True,bankstest))


# In[639]:


#true_rows

#false_rows

#info_gain(true_rows, false_rows,bankstest)


def best_split(data):
    data_list = np.array(data)
    optimal_gain = 0
    optimal_question = None
    no_of_columns=len(data_list[0]) - 1
    
    for column in range (no_of_columns): 
        values = set([row[column] for row in data_list])  
        
        for val in values: 
            ques = Question(column, val, data)
            true_values, false_values = partition(data, ques)
            tlength=len(true_values)
            flength=len(false_values)
            
            if tlength== 0 or  flength== 0:
                continue
                
            information_gain = info_gain(true_values, false_values,data_list)
            
            #Comparing so as to get the highest information gain as it means lowest entropy
            if information_gain>= optimal_gain:
                optimal_gain=information_gain
                optimal_question= ques
                
    return optimal_gain, optimal_question

#bgain, bquestion = best_split(bankstest)
#bquestion

#bankstest


def count(data):
    data_list = np.array(data)
    counts = {}
    for i in data_list:
        label = i[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] = counts[label]+1
    return counts


class leaf:
    def __init__(self,data):
        data_list = np.array(data)
        self.assumption = count(data)

class decision_node:
    def __init__(self,question,true,false):
        self.question = question
        self.true = true
        self.false = false


def print_tree(node):

    # Base case: we've reached a leaf
    if isinstance(node, leaf):
        print ("" + "Predict", node.assumption)
        return
    print (""+ str(node.question))
    print (""+ '--> True:')
    print_tree(node.true)
    print ("" + '--> False:')
    print_tree(node.false)

def decision_tree(data):
    data_list = np.array(data)
    information_gain, question = best_split(data)
    if information_gain == 0:
        return leaf(data)

    true_values, false_values = partition(data, question)
    #Building the true branch.
    true = decision_tree(true_values)

    #Building the false branch.
    false = decision_tree(false_values)

    return decision_node(question, true, false)


d_tree = decision_tree(banks)


def predict(data, node):

    data_list = np.array(data)
    # Base case: we've reached a leaf
    if isinstance(node, leaf):
        return node.assumption

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(data_list):
        return predict(data, node.true)
    else:
        return predict(data, node.false)


def printing_external(counts):
    c=counts.values()
    total = sum( c) * 1.0
    probabilities= {}
    for i in counts.keys():
        probabilities[i] = str(int(counts[i] / total * 100)) + "%"
    return probabilities



categories = []
def prediction_and_accuracy(data,tree):
    data_list = np.array(data)
    for i in data_list:
        #print ("Real value: %s. Predicted value: %s" %(i[-1], printing_external(predict(i, d_tree))))
        categories.append(printing_external(predict(i, tree)))
        
   # print(categories, "\n")  
    temp = []
    temp = [element for element in categories if len(element) == 1]
    t=len(temp)
    c=len(categories)
    tree_accuracy = (t/c ) * 100
    print("The  accuracy of data is",str(tree_accuracy) + "%")



def print_predictions(data,tree):
    categories2 = []
    data_list = np.array(data)
    for i in data_list:
        print ("Real value: %s. Predicted value: %s" %(i[-1], printing_external(predict(i, tree))))
    



def RandomForest(numberOfTrees, percentageOfAttributes): 
    forest = []
    total_attributes=len(banks.columns)-1 
    attr=int(percentageOfAttributes * (total_attributes)) 
    banks_nolabel=banks.drop('label',axis=1) 
    banks_test_list=np.array(bankstest)
    for i in range(numberOfTrees):
        training_dataset =banks_nolabel.sample(frac=percentageOfAttributes)
        decisionTree = decision_tree(training_dataset) 
        print_tree(decisionTree)
        #training_dataset1=np.array(training_dataset)
        #a= print_predictions(bankstest,decisionTree) 
        #a=predict(banks_test_list[0],decisionTree) 
        #predict(banks_list[0],d_tree) 
        #a=predict(banks_test_list[i],decisionTree)
        forest.append(decisionTree) 
    return forest


def RandomForestAccuracy(data,forest):
    data_list = np.array(data)
    a=[]
    b=prediction_and_accuracy(data_list,forest)
        


def function(data):
    data_list=np.array(data)
    for i in range(10):
        print("Number of Trees: " + str(5*i) + "Percentage Of Attributes: " +   str(0.1*i*100))
    f = RandomForest((5*i),(0.1*i))
    a=(RandomForestAccuracy(data,f[1]))
    return a


df= pd.DataFrame(function(banks_list))
gfg_csv_data = df.to_csv('predictions.csv', index = True)
