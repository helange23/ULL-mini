# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:03:55 2014

@author: henning
"""

import pickle
import numpy as np
from sklearn import mixture

class model_wrapper:
    
    def __init__(self):
        print 'Loading model data - this may take a while'
        self.GMM = pickle.load(open('GMM200','rb'))
        self.rep_vecs = pickle.load(open('rep_model200','rb'))
        self.embeddings = loadEmbeddings()
        self.root_weights = pickle.load(open('fixed_root','rb'))
        print 'Model data loaded'
        
        
    def getRootProb(self, argument):
        if self.embeddings.has_key(argument):
            probs = getGMMProbs(self.GMM, argument, self.embeddings)
            return np.dot(probs,self.root_weights)
        else:
            return 0
        
    def getProb(self, head, argument, val=0, direction='left'):
        if val==1 and direction == 'left':
            return getProb1(head, argument, self.GMM, self.rep_vecs[0], self.embeddings)
        if val==0 and direction == 'left':
            return getProb1(head, argument, self.GMM, self.rep_vecs[1], self.embeddings)
        if val==1 and direction == 'right':
            return getProb1(head, argument, self.GMM, self.rep_vecs[2], self.embeddings)
        if val==0 and direction == 'right':
            return getProb1(head, argument, self.GMM, self.rep_vecs[3], self.embeddings)
        else:
            print 'Something went wrong - cannot obtain a probability, returning 0'
            
            
    def findBestArgs(self, head, model=0):
        probs = list()
        for arg in self.embeddings.keys():
            probs.append(getProb1(head,arg,self.GMM, self.rep_vecs[model],self.embeddings))
            
        best = np.argsort(probs)[-10:]
        
        out = list()
        for b in best:
            out.append((self.embeddings.keys()[b],probs[b]))
        
        return out
        
        
        
        
def getGMMProbs(GMM,argument,embeddings):
    #logs = mixture.log_multivariate_normal_density(np.array([embeddings[argument]]), GMM.means_, GMM.covars_, GMM.covariance_type)[0] + np.log(GMM.weights_)
    
    X = np.array([embeddings[argument]])
    
    lpr = (mixture.log_multivariate_normal_density(X, GMM.means_, GMM.covars_,
                                               GMM.covariance_type))
               
    probs = np.exp(lpr)
    
    
    return probs[0]
    
    
    
def getProb1(head, argument, GMM, rep_vec, embeddings):
    probs = getGMMProbs(GMM,argument,embeddings)
    if rep_vec.has_key(head):
        return np.dot(probs,rep_vec[head])
    else:
        #print head, 'doesnt have this kind of dep, returning 0'
        return 0
        
def getProb2(head, argument, GMM, rep_vec, embeddings):
    #probs = getGMMProbs(GMM,argument,embeddings)
    if rep_vec.has_key(head):
        probs = GMM.predict_proba([embeddings[argument]])
    #print sum(probs)
    #print sum(rep_vec[head])
        return np.dot(probs,rep_vec[head])[0]
    else:
        return 0
        
        
        
        
def loadEmbeddings():
    #loads the word embeddings as a dictionary
    embeds = dict()    
    
    A = open('embeddings-scaled.EMBEDDING_SIZE=50.txt','r')
    for word in A:
        w = word.split()
        #normalizing to unit length
        embeds[w[0]] = np.double(w[1:])/np.linalg.norm(np.double(w[1:]))
        
    return embeds
    