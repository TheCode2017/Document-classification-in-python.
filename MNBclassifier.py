
# coding: utf-8

# In[4]:

import os
import random
from stop_words import get_stop_words

def getKeyWords(data_instance):
    wordList = data_instance.split(" ")
    return list(set(wordList))

#Randomly chooses 5 classes from the training set(To improve the running time of the algorithm)
def chooseRandomClass(path):
    fileList = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    fileList = random.sample(fileList, 5)
    return fileList

#Function that removes the stop words
def removeStopWords(text):
    stop_words = get_stop_words('en')
    words = getKeyWords(text)
    resultwords = [word for word in words if word.lower() not in stop_words]
    result = ' '.join(resultwords)
    result = ''.join(e for e in result if e.isalpha() or e == ' ')
    result = " ".join(result.split())
    return result

#Excludes the header portion of each file
def removeHeaders(lines):
    delimiter = 'Lines:'
    i = 0
    for i in range(0,len(lines)):
        if delimiter in lines[i]:
            break
    return ' '.join(lines[i+1:])

def readLinesFromFile(path):
    with open(path, 'r') as myfile:
        text = myfile.read()
    lines = text.split('\n')
    return lines

def getAttributesClasses(dirpath,dirlist):
    attr_list = []
    class_list = []
    for dir_name in dirlist:
        path_name = dirpath + dir_name + '/'
        for file in os.listdir(path_name):
            lines = readLinesFromFile(path_name + '/' + file)
            text = removeHeaders(lines)
            text = removeStopWords(text)
            text = removeStopWords(text)
            attr_list.append(text)
            class_list.append(dir_name)
    return attr_list,class_list

def probEvents(events):
    probabilites = dict()
    total_events = len(events)
    for i in range(0,len(events)):
        if probabilites.get(events[i]):
            probabilites[events[i]] += 1
        else:
            probabilites[events[i]] = 1
    for key in probabilites.keys():
        probabilites[key] /= total_events
    return probabilites

def getLivelihoodTable(data_instances,class_labels,prior):
    cond_prob = dict()
    total_data_instances = len(data_instances)
    for i in range(0, len(data_instances)):
        attributes = getKeyWords(data_instances[i])
        for j in range(0, len(attributes)):
            if cond_prob.get(attributes[j]):
                if cond_prob[attributes[j]].get(class_labels[i]):
                    cond_prob[attributes[j]][class_labels[i]] += 1
                else:
                    cond_prob[attributes[j]][class_labels[i]] = 1
            else:
                cond_prob[attributes[j]] = dict([(class_labels[i], 1)])
    for attr_key in cond_prob.keys():
        for attr_label_key in cond_prob[attr_key].keys() :
            cond_prob[attr_key][attr_label_key] /= (prior[attr_label_key]*total_data_instances)
    return cond_prob

def predict(data_instance,livelihood,prior,total_instances):
    attributes = getKeyWords(data_instance)
    max_posterior = -1
    max_posterior_label = ''
    for label in prior.keys() :
        posterior = prior[label]
        for i in range(0,len(attributes)):
            if livelihood.get(attributes[i]):
                if livelihood[attributes[i]].get(label):
                    posterior *= livelihood[attributes[i]][label]
                else:
                    posterior *= (1/((prior[label]*total_instances)+1))
            else:
                posterior *= (1 / ((prior[label] * total_instances) + 1))
        if posterior > max_posterior :
            max_posterior = posterior
            max_posterior_label = label
    return max_posterior_label

def getAccuracy(data_instances,class_labels,livelihood,prior):
    positive = 0
    total_instances = len(data_instances)
    for i in range (0,len(data_instances)):
        predicted_class = predict(data_instances[i], livelihood, prior,total_instances)
        if class_labels[i] == predicted_class :
            positive +=1
    return (positive/total_instances)*100

import sys

train_path = sys.argv[1]
test_path = sys.argv[2]
dir_list = chooseRandomClass(train_path)
print('Randomely Chosen classes : ',dir_list)
print('Parsing Train data...')
train_instances, train_class_labels = getAttributesClasses(train_path,dir_list)
print('Finding prior and livelihood...')
prior = probEvents(train_class_labels)
livelihood = getLivelihoodTable(train_instances,train_class_labels,prior)
print('Parsing Test data...')
test_instances, test_class_labels = getAttributesClasses(test_path,dir_list)
accuracy = getAccuracy(train_instances,train_class_labels,livelihood,prior)
print('Accuracy =',accuracy,'%')


# In[ ]:



