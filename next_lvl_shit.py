# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:31:32 2014

@author: henning
"""

import numpy as np
from sklearn import mixture, linear_model
import sys
#from scipy.stats import multivariate_normal
import pickle

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
	probs = getGMMProbs(GMM,argument,embeddings)
	#print sum(probs)
	#print sum(rep_vec[head])
	return np.dot(probs,rep_vec[head])


def createResponsibilityVector(dep, embeddings, GMM):
	reps = dict()
	for d in dep:
		if reps.has_key(d[0]):
			reps[d[0]] = reps[d[0]] + getGMMProbs(GMM, d[1], embeddings)
		else:
			reps[d[0]] = getGMMProbs(GMM, d[1], embeddings)

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
	print 'Size of rep_vecs', sys.getsizeof(rep_vecs)
	print 'Length', len(reps.keys())

	return reps


def initialize():
	embeddings = loadEmbeddings()
	all_deps = createDesignMatrix(loadCorpus()[0],embeddings)
	deleteUnusedWords(embeddings, all_deps)

	return embeddings, all_deps


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


def loadCorpus(size=10):
	#loads the corpus as a list of lists
	#every inner list denotes a sentence
	sentences = list()
	gold = list()
	A = open('test','r')
	s = list()
	g = list()
	i = 0
	for line in A:
		if line == '\n':
			i+= 1
			sentences.append(s)
			gold.append(g)
			s = list()
			g = list()
			if size == i:
				break
		else:
			ls = line.split()
			s.append([ls[1].replace('"',''), int(ls[6])-1])
			g.append(int(ls[6]))
	return sentences, gold


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


#
# rep_vecs = list()
# print 'Initializing'
# embeds, all_deps = initialize()
# print 'Embeddings and dependencies loaded, training GMM ...'
#
# g = getGMMClusters(embeds, 200)
# pickle.dump(g,open('GMM200','wb'))
#
# print 'GMM trained, creating rep vectors'
#
# for dep in all_deps:
# 	rep_vecs.append(createResponsibilityVector(dep, embeds, g))
# 	print 'Rep vectors created'
#
# print 'Saving'
#
# pickle.dump(rep_vecs,open('rep_model200','wb'))
# print 'Done'
#