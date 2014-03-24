from pprint import pprint
import numpy as np
# from nltk.parse.pchart import InsideChartParser as Parser
from nltk.parse.viterbi import ViterbiParser as Parser
from nltk.grammar import ContextFreeGrammar
from nltk.grammar import Production
from nltk.grammar import WeightedProduction
from nltk.grammar import induce_pcfg
from nltk import Nonterminal
from model_wrapper import model_wrapper
from next_lvl_shit import loadCorpus
from next_lvl_shit import getSentencesWithKnownWords
from nltk.tree import Tree
import re


def getProb(head, arg):
	"""
	TODO: Needs to fetch probabilities from the model.
	@param head: Head word
	@param arg: Potential argument of head word.
	@return: Probability of attaching arg to head.
	"""
	return np.random.rand(1,1)[0][0]

def getGrammar(sentence):
	"""
	Constructs an ad-hoc split head DMV grammar for the given sentence.
	@param sentence: Input sentence as a list of tokens
	@return: NLTK grammar with weighted productions.
	"""
	productions = []
	S = Nonterminal('S')
	for i, head in enumerate(sentence):
		Y_head = Nonterminal('Y_'+head)
		L_head = Nonterminal('L_'+head)
		R_head = Nonterminal('R_'+head)
		L1_head = Nonterminal('L1_'+head)
		R1_head = Nonterminal('R1_'+head)
		LP_head = Nonterminal('LP_'+head)
		RP_head = Nonterminal('RP_'+head)
		productions.append(Production(Y_head, [L_head, R_head]))
		productions.append(Production(L_head, [head+'_l']))
		productions.append(Production(R_head, [head+'_r']))
		productions.append(Production(L_head, [L1_head]))
		productions.append(Production(R_head, [R1_head]))
		productions.append(Production(LP_head, [head+'_l']))
		productions.append(Production(RP_head, [head+'_r']))
		productions.append(Production(LP_head, [L1_head]))
		productions.append(Production(RP_head, [R1_head]))
	grammar = induce_pcfg(Nonterminal('S'), productions)
	for i, head in enumerate(sentence):
		L1_head = Nonterminal('L1_'+head)
		R1_head = Nonterminal('R1_'+head)
		LP_head = Nonterminal('LP_'+head)
		RP_head = Nonterminal('RP_'+head)
		Y_head = Nonterminal('Y_'+head)

		grammar.productions().append(WeightedProduction(S, [Y_head], prob=model.getRootProb(head)))
		for j in xrange(0, i):
			arg = sentence[j]
			prob = model.getProb(head, arg, direction='left')

			grammar.productions().append(WeightedProduction(L1_head, [Nonterminal('Y_'+sentence[j]), LP_head],prob=prob))
		for j in xrange(i+1, len(sentence)):
			arg = sentence[j]
			prob = model.getProb(head, arg, direction='right')
			grammar.productions().append(WeightedProduction(R1_head, [RP_head, Nonterminal('Y_'+sentence[j])], prob=prob))

	return grammar
			
def splitSentence(sentence):
	"""
	Converts sentence to split head structure
	@param sentence: Input sentence as a list of tokens
	@return: List of split head input sentence tokens.
	"""
	res = []
	for word in sentence:
		res.append(word + "_l")
		res.append(word + "_r")
	return res

def run_parser(corpus, n=1000):
	"""
	Runs the parser on a corpus.
	@param corpus: List of lists with input tokens
	"""
	total = 0
	total_sent = 0
	right = 0
	baseline = 0
	for item in corpus:
		sentence = [word[0] for word in item]
		grammar = getGrammar(sentence)
		parser = Parser(grammar)
		sent = splitSentence(sentence)
		tree = parser.parse(sent)
		# tree.draw()
		# print tree.pprint(margin=30)
		deps = extractDepParse(tree, sentence)
		print len(deps), len(sentence)
		for i in xrange(0, len(sentence)):
			total += 1
			print deps[i], item[i][1]+1
			if deps[i] == item[i][1]+1:
				right += 1
			if i == item[i][1]+1:
				baseline += 1
		# cyk(sent)
		total_sent += 1
		if total_sent == n:
			break
	P = ((right*1.0)/total) * 100.0
	R = 100.0
	print "baseline ", ((baseline*1.0)/total) * 100.0
	print "precision", P
	print "f1 ", (2*P*R)/(P+R)
	return P

def extractDepParse(tree, sentence):
	"""
	Transforms a PCFG parsed tree in DMV split head encoding back to a dependency definition.
	Has issues with the same word occurring twice in the sentence :/
	@param tree: NLTK tree
	@param sentence: The original input sentence
	"""
	tree = tree.pprint(margin=30)
	deps = [None for i in xrange(0, len(sentence))]
	already_assigned = {}
	stack = []
	# I could write a comment explaining this code, but this code is very dumb and probably wrong so I don't expect you to understand
	for line in tree.split('\n'):
		# indents = len(re.findall('[ ]*', line)[0])
		if line.strip()[0:3] == '(S':
			stack.append(0)
		if line.strip()[2:4] == '1_':
			dep = line.strip()[4:]
			dep_index = sentence.index(dep) + 1
			stack.append(dep_index)
		if line.strip()[0:3] == '(Y_' and len(stack) > 0:
			y = line.strip()[3:]
			dep_index = stack.pop()
			y_index = sentence.index(y)
			if already_assigned.has_key(y_index):
				y_index = sentence.index(y, y_index+1)
			else:
				already_assigned[y_index] = True

			deps[y_index] = dep_index

	for i, word in enumerate(sentence):
		print word, deps[i]
	print "\n"
	return deps


def cyk(sentence):
	"""
	Abandoned CYK parser, ignore.

	@param sentence:
	@return:
	"""
	n = len(sentence)
	# chart = [[{}]]
	chart = [[{}]*(n+1)]*(n+1)
	for i in range(0, n):
		chart[i][i+1][sentence[i]] = 1

	for max in range(1, n):
		for min in reversed(range(0, max-1)):
			for mid in range(min+1, max):
				return ""
				# chart[min][mid][]


# sents = ["The big dog barks to this other dog".split(" "),
# sents = ["The new rate will be payable".split(" ")]
model = model_wrapper(k=500)
sents = loadCorpus()
sents = getSentencesWithKnownWords(sents, model.embeddings)
run_parser(sents)

