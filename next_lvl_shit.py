# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:31:32 2014

@author: henning
"""

import numpy as np
from sklearn import mixture
import pickle
import matplotlib.pyplot as plt

def getGMMClusters(embeddings, k=50):
    g = mixture.GMM(n_components=k,covariance_type='tied')
    g.fit(embeddings.values())
    
    return g
    
def loadGMMClusters():
    return pickle.load(open('gmm_clusters','rb'))
    
def findBestArgs(head, GMM, rep_vec, embeddings):
    probs = list()
    for arg in embeddings.keys():
        probs.append(getProb(head,arg,GMM,rep_vec,embeddings))
        
    best = np.argsort(probs)[-10:]
    
    out = list()
    for b in best:
        out.append((embeddings.keys()[b],probs[b]))
    
    return out
        
def getGMMProbs(GMM,argument,embeddings):
    #logs = mixture.log_multivariate_normal_density(np.array([embeddings[argument]]), GMM.means_, GMM.covars_, GMM.covariance_type)[0] + np.log(GMM.weights_)
    
    X = np.array([embeddings[argument]])
    
    lpr = (mixture.log_multivariate_normal_density(X, GMM.means_, GMM.covars_,
                                               GMM.covariance_type))
               
    probs = np.exp(lpr)
    
    
    return probs[0]
    
    
    
def getProb(head, argument, GMM, rep_vec, embeddings):
    #probs = getGMMProbs(GMM,argument,embeddings)
    probs = GMM.predict_proba([embeddings[argument]])
    #print sum(probs)
    #print sum(rep_vec[head])
    return np.dot(probs,rep_vec[head])[0]
    
def trainRoot(root, embeddings, GMM):
    out = getGMMProbs(GMM, root[0], embeddings)
    #del root[0]
    for arg in root:
        out = out + GMM.predict_proba([embeddings[arg]])
        
    return out/sum(out)
    
    
def createResponsibilityVector(dep, embeddings, GMM):
    reps = dict()
    for d in dep:
        if reps.has_key(d[0]):
            reps[d[0]] = reps[d[0]] + GMM.predict_proba([embeddings[d[1]]])[0]
        else:
            reps[d[0]] = GMM.predict_proba([embeddings[d[1]]])[0]
            
    for head in reps.keys():
        reps[head]=reps[head]/sum(reps[head])
    
    '''
    h_a = dict()
    for d in dep:
        if h_a.has_key(d[0]):
            h_a[d[0]].append(d[1])
        else:
            h_a[d[0]]=list()
            h_a[d[0]].append(d[1])
    
    rep_vecs = dict()
    print 'length', len(h_a.keys())
    for w in h_a.keys():
        #regress_input = list()
        test = np.zeros((GMM.n_components,1))
        for arg in h_a[w]:
            #regress_input.append(getGMMProbs(GMM, arg, embeddings))
            test = test + getGMMProbs(GMM, arg, embeddings)
            
        #clf = linear_model.Lasso(positive=True,alpha=0.0,normalize=False, fit_intercept=False)
        #print np.array(regress_input).shape
        #clf.fit(np.array(regress_input), np.ones((len(regress_input),1)))

        #print test/sum(test)
        rep_vecs[w]=test/sum(test)
    '''
    
    #print 'Size of h_a', sys.getsizeof(h_a)
    #print 'Size of rep_vecs', sys.getsizeof(rep_vecs)
    #del h_a
            
    return reps
    
    
def initialize():
    embeddings = loadEmbeddings()
    all_deps, root = createDesignMatrix(loadCorpus(),embeddings)
    deleteUnusedWords(embeddings, all_deps, root)
    
    return embeddings, all_deps, root
    
    
def createDesignMatrix(corpus, embeddings):
    
    leftstop = list()
    leftgo = list()
    rightstop = list()
    rightgo = list()
    root = list()
    
    good = 0
    bad = 0

    
    for sentence in corpus:
        for i in xrange(0,len(sentence)):
            dependency = sentence[i][1]
            if not dependency == -1:
            
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
                if embeddings.has_key(sentence[i][0]):
                    root.append(sentence[i][0])

            
    #if for both words in a dependency pair there are embedding then it is a
    #good pair otherwise a bad one
    print 'bad',bad,'good',good

    return [leftstop, leftgo, rightstop, rightgo], root
            
        
        
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
    
    
def deleteUnusedWords(embeddings, all_deps, root):
    unique_words = set()
    for dep in all_deps:
        for d in dep:
            unique_words.add(d[0])
            unique_words.add(d[1])
    for w in root:
        unique_words.add(w)
    for k in embeddings.keys():
        if not k in unique_words:
            del embeddings[k]
            
def getMostProbRoots(root_weights, GMM, embeddings):
    probs = list()
    for argument in embeddings.keys():
        prob = getGMMProbs(GMM, argument, embeddings)
        prob = np.dot(prob,root_weights)
        probs.append(prob)
        
    best = np.argsort(probs)[-10:]
    
    print best
    
    out = list()
    for b in best:
        out.append((embeddings.keys()[b],probs[b]))
    
    return out
    
def visualizeCluster(rep_vec):
    r = np.zeros((len(rep_vec[rep_vec.keys()[0]]),1))
    for head in rep_vec.keys():
        r = r + rep_vec[head]
        
    return r[0]/len(rep_vec.keys())
    
def visualizeClusters(rep_vecs, root_weights, GMM):
    plt.plot(visualizeCluster(rep_vecs[0]),label='Left stop')
    plt.plot(visualizeCluster(rep_vecs[1]),label='Left go')
    plt.plot(visualizeCluster(rep_vecs[2]),label='Right stop')
    plt.plot(visualizeCluster(rep_vecs[3]),label='Right go')
    plt.plot(GMM.weights_, label='GMM weights')
    plt.plot(root_weights,label='Root')
    plt.legend()
    
    
def getWordsInCluster(c, GMM, embeds):
    out = list()
    for k in embeds.keys():
        if GMM.predict([embeds[k]]) == c:
            out.append(k)
            
    return out
    
def visualizeRoots(root, root_weights, GMM, embeds):
    word_hist = np.zeros(root_weights.size)
    soft = np.zeros(root_weights.size)
    for w in root:
        soft = soft + GMM.predict_proba([embeds[w]])
        word_hist[GMM.predict([embeds[w]])] = word_hist[GMM.predict([embeds[w]])] + 1
    word_hist = word_hist/len(root)
    soft = soft/sum(soft)
    
    plt.plot(soft, label='Soft')
    plt.plot(word_hist, label='True word cluster probs')
    plt.plot(root_weights, label='Root weights')
    plt.legend
    return word_hist
    
def getSentencesWithKnownWords(corpus, embeddings):
    out = list()
    for sentence in corpus:
        known_all = True
        for dep in sentence:
            if not embeddings.has_key(dep[0]):
                known_all = False
        if known_all and len(sentence)<10:
            out.append(sentence)
    return out
    
def printAllArguments(head, dep):
    for d in dep:
        if d[0] == head:
            print d[1]

# rep_vecs = list()
# print 'Initializing'
# embeds, all_deps, root = initialize()
#
# print 'Embeddings and dependencies loaded, training GMM ...'
#
# k=500
# g = getGMMClusters(embeds, k)
# pickle.dump(g,open('GMM'+str(k),'wb'))
# g = pickle.load(open('GMM'+str(k),'rb'))
#
# print 'GMM trained, training root'
# root_weights = trainRoot(root, embeds, g)
# pickle.dump(root_weights,open('root'+str(k),'wb'))
#
# print 'root trained, training rep vectors'
#
# for dep in all_deps:
#     rep_vecs.append(createResponsibilityVector(dep, embeds, g))
#     print 'Rep vectors created'
#
# print 'Saving'
#
# pickle.dump(rep_vecs,open('rep_model'+str(k),'wb'))
# print 'Done'