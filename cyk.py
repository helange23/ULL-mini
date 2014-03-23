from pprint import pprint
import numpy as np
# from nltk.parse.pchart import InsideChartParser as Parser
from nltk.parse.viterbi import ViterbiParser as Parser
from nltk.grammar import ContextFreeGrammar
from nltk.grammar import Production
from nltk.grammar import WeightedProduction
from nltk.grammar import induce_pcfg
from nltk import Nonterminal
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
	for i, head in enumerate(sentence):
		S = Nonterminal('S')
		Y_head = Nonterminal('Y_'+head)
		L_head = Nonterminal('L_'+head)
		R_head = Nonterminal('R_'+head)
		L1_head = Nonterminal('L1_'+head)
		R1_head = Nonterminal('R1_'+head)
		LP_head = Nonterminal('LP_'+head)
		RP_head = Nonterminal('RP_'+head)
		productions.append(Production(S, [Y_head]))
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
		for j in xrange(0, i):
			arg = sentence[j]
			prob = getProb(head, arg)

			grammar.productions().append(WeightedProduction(L1_head, [Nonterminal('Y_'+sentence[j]), LP_head],prob=prob))
		for j in xrange(i+1, len(sentence)):
			arg = sentence[j]
			prob = getProb(head, arg)
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

def run_parser(corpus):
	"""
	Runs the parser on a corpus.
	@param corpus: List of lists with input tokens
	"""
	for sentence in corpus:
		grammar = getGrammar(sentence)
		parser = Parser(grammar)
		sent = splitSentence(sentence)
		tree = parser.parse(sent)
		tree.draw()
		# print tree.pprint(margin=30)
		extractDepParse(tree, sentence)

		# cyk(sent)

def extractDepParse(tree, sentence):
	"""
	Transforms a PCFG parsed tree in DMV split head encoding back to a dependency definition
	@param tree: NLTK tree
	@param sentence: The original input sentence
	"""
	tree = tree.pprint(margin=30)
	deps = [None for i in xrange(0, len(sentence))]
	last_index = {}
	stack = []
	# I could write a comment explaining this code, but this code is very dumb and probably wrong so I don't expect you to understand
	for line in tree.split('\n'):
		# indents = len(re.findall('[ ]*', line)[0])
		if line.strip()[2:4] == '1_':
			stack.append(line.strip()[4:])
		if line.strip()[0:3] == '(Y_' and len(stack) > 0:
			y = re.findall('Y_[^ ]*', line)[0][2:].strip()
			dep = stack.pop()
			if not last_index.has_key(y):
				last_index[y] = 0
			if not last_index.has_key(dep):
				last_index[dep] = 0
			dep_index = sentence.index(dep, last_index[dep])
			y_index = sentence.index(y, last_index[y])
			deps[y_index] = dep_index + 1
			last_index[dep] = dep_index
			last_index[y] = y_index

	for i, word in enumerate(sentence):
		print word, deps[i]
	print "\n"


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


sents = ["The big dog barks to this other dog".split(" "),
		 "John thinks Mary is not quite right in the head".split(" ")]
run_parser(sents)

