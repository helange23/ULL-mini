# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:36:17 2014

@author: henning
"""

import numpy as np
from sklearn import linear_model

#this is a test

def trainLinearModel(data, embeddings):
    
    input1,output1 = unzipData(data, embeddings)
    
    input1 = np.array(input1)
    output1 = np.array(output1)
    
    print input1.shape,output1.shape

    clf = linear_model.LinearRegression(normalize=True)
    clf.fit(input1, output1)
    
    return clf
    
def unzipData(data, embeddings):
    input1 = list()
    output1 = list()
    
    #unzipping the data
    for d in data:
        input1.append(np.append(embeddings[d[0]],1))
        output1.append(embeddings[d[1]])
    
    return input1,output1
    
def getModelError(data, model, embeddings):
    
    input1, output1 = unzipData(data, embeddings)
    
    return model.score(input1, output1)
    
def getAllErrors(datas, models, embeddings):
    for i in xrange(0,len(datas)):
        print 'Error :',getModelError(datas[i],models[i],embeddings)

        
def trainABit(all_deps, embeds, models):
    for i in xrange(0,200):
        embeds, models = improveEmbeddings(all_deps,embeds,models)
        print 'number',i
    return embeds, models
    
def improveEmbeddings(all_dependencies, embeddings, all_models=None):
    #initial models
    if all_models == None:
        all_models = list()
        for d in all_dependencies:
            all_models.append(trainLinearModel(d, embeddings))

    #learning rate
    alpha = 0.05
    
    d_embed = dict()
    
    for i in xrange(0,4):
        dep = all_dependencies[i]
        model = all_models[i]
        
        
        
        for d in dep:
            desired_arg = model.predict(np.append(embeddings[d[0]],1))
            if d_embed.has_key(d[1]):
                d_embed[d[1]] = d_embed[d[1]] + (np.array(desired_arg)-np.array(embeddings[d[1]]))
            else:
                d_embed[d[1]] = (np.array(desired_arg)-np.array(embeddings[d[1]]))
            #renormalize to unit length
            #embeddings[d[1]] = embeddings[d[1]]/np.linalg.norm(embeddings[d[1]])
        
    for k in d_embed.keys():
        d = alpha*np.sqrt(np.linalg.norm(d_embed[k]))
        embeddings[k] = embeddings[k]+d*d_embed[k]
        embeddings[k] = embeddings[k]/np.linalg.norm(embeddings[k])
    
    print np.linalg.norm(d_embed['book'])
        
    for i in xrange(0,4):
        all_models[i] = trainLinearModel(all_dependencies[i], embeddings)
        
        
    return embeddings, all_models
            
    
    
def findClosestArgument(word, model, embeddings):
    argument = model.predict(np.append(embeddings[word],1))
    print np.linalg.norm(argument)
    argument = argument/np.linalg.norm(argument)

    matrix = embeddings.values()

    listing = np.dot(matrix,argument)
    out = list()

    best = np.argsort(listing)[-10:]
    for b in best:
        out.append(embeddings.keys()[b])
    
    return out

def getMostFreqWords(all_deps):
    counts = dict()
    for dep in all_deps:
        for d in dep:
            if counts.has_key(d[1]):
                counts[d[1]]=counts[d[1]]+1
            else:
                counts[d[1]]=1
    most_freq = np.argsort(counts.values())[-700:]
    
    out = set()    
    
    for m in most_freq:
        out.add(counts.keys()[m])
        print counts.keys()[m], counts[counts.keys()[m]]
        
    return out
        

    
def conditionalError(word, argument, model, embeddings):
    argument_predict = model.predict(np.append(embeddings[word],1))
    argument_predict = argument_predict/np.linalg.norm(argument_predict)
    
    return np.exp(-np.linalg.norm(np.array(embeddings[argument])-np.array(argument_predict)))


def createDesignMatrix(corpus, embeddings):
    
    leftstop = list()
    leftgo = list()
    rightstop = list()
    rightgo = list()
    
    good = 0
    bad = 0

    
    for sentence in corpus:
        for i in xrange(0,len(sentence)):
            dependency = sentence[i][1]            
            
            #make sure that both words have embeddings
            if(embeddings.has_key(sentence[i][0]) and embeddings.has_key(sentence[dependency][0])):
                good = good + 1
                
                #left dependency
                if i < dependency:
                    if isFirstLeftDependent(sentence,i):
                        leftstop.append([sentence[dependency][0], sentence[i][0]])
                    else:
                        leftgo.append([sentence[dependency][0], sentence[i][0]])
                #right dependency
                else:
                    if isLastRightDependent(sentence,i):
                        rightstop.append([sentence[dependency][0], sentence[i][0]])
                    else:
                        rightgo.append([sentence[dependency][0], sentence[i][0]])
                
            else:
                bad = bad + 1

            
    #if for both words in a dependency pair there are embedding then it is a
    #good pair otherwise a bad one
    print 'bad',bad,'good',good

    return [leftstop, leftgo, rightstop, rightgo]
            
        
        
def isFirstLeftDependent(sentence, idx):
    dependency = sentence[idx][1]
    for i in xrange(0,dependency):
        if sentence[i][1] == dependency and i < idx:
            return False
    return True
    
def isLastRightDependent(sentence, idx):
    dependency = sentence[idx][1]
    for i in xrange(dependency,len(sentence)):
        if sentence[i][1] == dependency and i > idx:
            return False
    return True


def loadCorpus():
    #loads the corpus as a list of lists
    #every inner list denotes a sentence
    sentences = list()
    A = open('test','r')
    s = list()
    for line in A:
        if line == '\n':
            sentences.append(s)
            s = list()
        else:
            ls = line.split()
            s.append([ls[1].replace('"',''), int(ls[6])-1])
    return sentences


def loadEmbeddings():
    #loads the word embeddings as a dictionary
    embeds = dict()    
    
    A = open('embeddings-scaled.EMBEDDING_SIZE=50.txt','r')
    for word in A:
        w = word.split()
        #normalizing to unit length
        embeds[w[0]] = np.double(w[1:])/np.linalg.norm(np.double(w[1:]))
        
    return embeds
    
    
def deleteUnusedWords(embeddings, all_deps):
    unique_words = set()
    for dep in all_deps:
        for d in dep:
            unique_words.add(d[0])
            unique_words.add(d[1])
    for k in embeddings.keys():
        if not k in unique_words:
            del embeddings[k]

def initialize():
    embeddings = loadEmbeddings()
    all_deps = createDesignMatrix(loadCorpus(),embeddings)
    deleteUnusedWords(embeddings, all_deps)
    
    return embeddings, all_deps
    
    
    

def prepareForSne(all_deps, embeddings):
    
    words = getMostFreqWords(all_deps)
    
    f = open('sne.txt','w')

    for w in words:
        stra = ' '.join(map(str, embeddings[w]))
        print >>f, w, stra