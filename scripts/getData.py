
# coding: utf-8

# In[16]:

import csv
import sys

# Change working path to read dataset
import os
os.chdir("../dataset/")


# In[ ]:

# Make label of 1, 2, 3, 4.... into a hot vector 
# whose ith index is the label and the rest are 0
def make_hot(lables):
    for i in range(len(lables)):
        temp = [0.0] * 10
        temp[int(lables[i])] = 1.0
        lables[i] = temp
    return lables


# In[ ]:

def normalize(myList):
    for i in range(len(myList)):
        myList[i] = [k/255 for k in myList[i]] 
    return myList


# In[10]:

def getTrain(train_num):
    train_x = []
    train_y = []
    
    f = open('train.csv', 'rt')
    if_first = True 
    index = 0
    try:
        reader = csv.reader(f)
        for row in reader:
            if index > train_num:
                break
            
            index = index + 1
            # Get rid of the label row 
            if if_first == True:
                if_first = False
                continue
            train_y.append(row[0])
            train_x.append(map(float, row[1:len(row)]))
    finally:
        f.close()

    train_x = normalize(train_x)
    
    # Convert train_y value into a hot vector of 10 * 1
    # Make the probability of the true label 1
    train_y = make_hot(train_y)
        
    return train_x,train_y


# In[9]:

def getTest(test_min, test_max):
    test_x = []
    test_y = []
    f = open('train.csv', 'rt')
    if_first = True 
    index = 0

    try:
        reader = csv.reader(f)
        for row in reader:
            index = index + 1
            if index > test_min and index < test_max:
                # Get rid of the label row 
                if if_first == True:
                    if_first = False
                    continue
                test_y.append(row[0])
                test_x.append(map(float, row[1:len(row)]))
    finally:
        f.close()
        
    test_x = normalize(test_x)
    test_y = make_hot(test_y)
    
    return test_x, test_y

