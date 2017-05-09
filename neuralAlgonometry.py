import sys
# path to latex2sympy. Converts a latex equation to sympy format
sys.path.append('/Users/Forough/Documents/bitBucket/math-knowledge-base/Codes/latex2sympy')
import json
from sympy import *
import pprint
# from process_latex import process_sympy
import re
# from prover import parseEquation
import copy
from itertools import count
import random
import compiler
import mxnet as mx
import numpy as np
# from nnTreeUtils import nnTree, BucketEqIterator
from nnTreeMain import nnTree, BucketEqIterator


################################################################################
# global recDepth
# global globalCtr
# global maxDepth

functionOneInp = ['sin', 'cos', 'csc', 'sec', 'tan', 'cot',
				  'asin', 'acos', 'acsc', 'asec', 'atan', 'acot',
				  'sinh', 'cosh', 'csch', 'sech', 'tanh', 'coth',
				  'asinh', 'acosh', 'acsch', 'asech', 'atanh', 'acoth',
				  'exp']# , 'IdentityFunction', 'root', 'sqrt'
functionOneInpSet = set(functionOneInp)

functionTwoInp = ['Equality', 'Add', 'Mul', 'Pow', 'log']#'Min', 'Max','atan2', 'Div'
functionTwoInpSet = set(functionTwoInp)

functionBothSides = ['Add', 'Mul', 'Pow', 'log'] # Anything else?
functionBothSidesSet = set(functionBothSides)

variables = ['Symbol']
variablesSet = set(variables)

consts = ['NegativeOne', 'NaN', 'Infinity', 'Exp1', 'Pi', 'One',
          'Half', 'Integer', 'Rational', 'Float']#, 'Float'
constsSet = set(consts)

intList = [0, 2, 3, 4, 10]
intSet = set(intList)

ratList = [2/5]
ratSet = set(ratList)

floatList = [0.7]
floatSet = set(floatList)

################################################################################
functionVocab = []
tmp = []
functionVocab.extend(functionOneInp)
functionVocab.extend(functionTwoInp)
functionVocab.extend(functionBothSides) 
tmp.extend(functionVocab)
tmp.extend(variables) 
tmp.extend(consts)

functionDictionary = {}
ctr = 1
for f in tmp:
	functionDictionary[f] = ctr
	ctr+=1
################################################################################
# equation functions: #
def readJson(path, equations, ranges, jsonAtts):
	with open(path) as data_file:    
		data = json.load(data_file)
	for i in range(0,len(data)):
		eq = data[i]["equation"]
		# print "eq before:", eq
		while len(eq)>0 and (eq[-1]=='.' or eq[-1]==',' or eq[-1]=='!' or eq[-1]=='\\'):
			# print "here"
			eq = eq[:-1]
		eq = str(eq.encode("utf-8"))
		# print "eq after:", eq
		equations.append(eq)
		ranges.append(data[i]["range"])

def writeJson(path, equations, ranges, variables, labels):
	data = []
	for i, eq in enumerate(equations):
		d = {}
		p = preOrder(eq)
		d['equation'] = {'func':p[0][:-1], 'vars': p[1][:-1], 'nodeNum':p[2][:-1],
		                 'numNodes':str(eq.numNodes), 'variables':variables[i]}
		# d['vars'] = p[1][:-1]
		# d['nodeNum'] = p[2][:-1]
		d['label'] = str(int(labels[i].asnumpy()[0]))
		# d['ranges'] = preOrder(ranges[i])
		data.append(d)

	with open(path, 'w') as outfile:
		json.dump(data, outfile, indent=4)

def readJsonEquations(path):
	equations = []
	variables = []
	ranges = []
	labels = []

	with open(path) as data_file:
		data = json.load(data_file)
	for i in range(len(data)):
		d = data[i]
		currFunc      = d['equation']['func'].split(',')
		currVarname   = d['equation']['vars'].split(',')
		currNumber    = d['equation']['nodeNum'].split(',')
		currVariables = d['equation']['variables']
		currNumNodes  = int(d['equation']['numNodes'])
		currLabel     = mx.nd.array([int(d['label'])], dtype='float32')
		# currRange = d['range'].split(',')

		currEq = makeEqTree(currFunc, currVarname, currNumber, currNumNodes)
		equations.append(currEq)
		# ranges.append(rangeTree)
		labels.append(currLabel)
		variables.append(currVariables)

	return [equations, variables, ranges, labels]

def makeEqTree(func, varname, number, numNodes=-1):
	if len(func) == 0:
		return
	if func[0] != '#':
		eq = EquationTree(func=func[0], varname=varname[0], number=int(number[0]), args=[])
		if numNodes != -1:
			eq.numNodes = numNodes
		del func[0]
		del varname[0]
		del number[0]
		eq.args.append(makeEqTree(func, varname, number))
		eq.args.append(makeEqTree(func, varname, number))
		if eq.args[1] == None:
			del eq.args[1]
		if eq.args[0] == None:
			del eq.args[0]
		return eq
	else:
		del func[0]
		del varname[0]
		del number[0]
		return

 
def readAxioms(path, axioms, axiomVariables):
	f = open(path, 'r')
	for axiom in f:
		axiomTree = compiler.parse(axiom)
		for i in range(0,3):
			# print axiomTree
			axiomTree = axiomTree.getChildNodes()[0]
		axioms.append(axiomTree)

		#extracting variables: Pi will also be a variable resolve this later
		extractedVars = re.findall("Name\(.*?\)", str(axiomTree))
		var = []
		for v in extractedVars:
			vsplit = v.split('\'')
			var.append(vsplit[1])
		# removing repetitions:
		var = set(var)
		var = list(var)
		varDict = {'%s'%(e):i for i, e in enumerate(var)}
		axiomVariables.append(varDict)

def parseEquation(equations, parsedEquations, variabs):
	for ctr in range(len(equations)):
		# eq = sympify(equations[ctr], evaluate=False)
		eq = process_sympy(equations[ctr])
		# print equations[ctr]
		# eq = sympify(equations[ctr], evaluate=False)
		# eq = sympify(equations[ctr])
		var = eq.free_symbols
		varDict = {'%s'%(e):i for i, e in enumerate(var)}

		parsedEquations.append(eq)
		# print variables
		variabs.append(varDict)

def parseRange(ranges, parsedRanges, variabs):
	for i, rng in enumerate(ranges):
		d = {}
		for key, val in rng.iteritems():
			if key not in variabs[i]:
				raise AssertionError('unknown variable %s. Not in the variable dictionary %s' %(key, variabs))
			val = val.split(',')
			assert len(val) <= 2, 'cannot have more than two inputs to range'
			val = [compiler.parse(val[n]) for n in range(len(val))]
			for j in range(len(val)):
				for k in range(0,3):
					val[j] = val[j].getChildNodes()[0]
			d[key] = val
		parsedRanges.append(d)

def preOrder(equation):
	funcStr = equation.func + ','
	varStr = equation.varname + ','
	numStr = str(equation.number) + ','
	if len(equation.args) == 0:
		funcStr = funcStr + '#,#,'
		varStr = varStr + '#,#,'
		numStr = numStr + '#,#,'
	elif len(equation.args) == 1:
		p = preOrder(equation.args[0])
		funcStr = funcStr + p[0] + '#,'
		varStr = varStr + p[1] + '#,'
		numStr = numStr + p[2] + '#,'
	elif len(equation.args) == 2:
		p0 = preOrder(equation.args[0])
		p1 = preOrder(equation.args[1])
		funcStr = funcStr + p0[0] + p1[0]
		varStr = varStr + p0[1] + p1[1]
		numStr = numStr + p0[2] + p1[2]

	return [funcStr, varStr, numStr]

def buildEq(parsedEquation, variables):
	# enforces the tree to be binary regardless of the input equation's structure. This
	# allows us to do weight sharing as well as incorporating for commutative or associative 
	# propoerties of addition and multiplication
	def expand(parent, children):
		# always exapnds in the same order. Does not consider commutativity or associativity
		if len(children) > 2:
			par = EquationTree(func=parent, args=[buildEq(children[0], variables), expand(parent=parent, children=children[1:])])
		else:
			par = EquationTree(func=parent, args=[buildEq(arg, variables) for arg in children])

		return par

	# extracting the function name from the sympy parse
	func = str(parsedEquation.func)
	func = func.split('.')[-1]
	while func[-1]=='\'' or func[-1]=='>':
		func = func[:-1]

	# pre-order tree generation
	# main traversal:
	if len(parsedEquation.args) > 2:
		root = EquationTree(func=func, args=[buildEq(parsedEquation.args[0], variables), expand(parent=func, children=parsedEquation.args[1:])])
	else:
		root = EquationTree(func=func, args=[buildEq(arg, variables) for arg in parsedEquation.args])
	# base case: leaf
	if len(parsedEquation.args)==0:
		key = '%s'%(parsedEquation)
		# print key
		if key in variables:
			if key == 'pi' or key == 'Pi':
				root.func = 'Pi'
				root.varname = 'pi'
			else:
				root.varname = 'var_%d' %(variables[key])
		else:
			# note: if we do not do str, it will return the sympy class in the varname
			# This could come in handy anyways. I might need to add a node attribute that carries this information.
			# also note: for pi, we might want to handle it with hard coding for now.
			root.varname = str(parsedEquation)
	return root

def buildAxiom(axiomTree, variables):

	# pre-processing function to be compatible with the sympy terminology
	# This is a temporary hack. Might need to come back to this and make it smarter
	# too much hand coding already :/
	negOneFlag = 0

	func = str(axiomTree)
	func = func.split('(')[0]
	if func == 'Power':
		func = 'Pow'
	elif func == 'Const' and axiomTree.getChildren()[0] == 1:
		# print "here: functionality is One"
		func = 'One'
	elif func == 'Const' and axiomTree.getChildren()[0] == -1:
		func = 'NegativeOne'
	elif func == 'Const':
		func = 'Integer'
	elif func == 'Name':
		if str(axiomTree.getChildren()[0]) == 'Half':
			func = 'Half'
		else: func = 'Symbol'
	elif func == 'UnarySub':
		child = axiomTree.getChildren()[0]
		childFunc = str(child)
		childFunc = childFunc.split('(')[0]
		if childFunc == 'Const':
			childFunc = 'Integer'
		elif childFunc == 'Name':
			childFunc = 'Symbol'
		
		if child.getChildren()[0] == 1:
			func = 'NegativeOne'
			negOneFlag = 1
			axiomTree = axiomTree.getChildren()[0]
		else:
			func = 'Mul'
			return EquationTree(func=func, 
								args=[EquationTree(func='NegativeOne', args=[], varname=str(-1)), 
								EquationTree(func=childFunc, args=[], varname=str(child.getChildren()[0]))])
	# else:
	# 	print "I have found an undefined category:", func, axiomTree.getChildren()[0]

	children = axiomTree.getChildren()
	if len(children) == 1 and (isinstance(children[0], int) or isinstance(children[0], str)):
		# print "in leaf"
		if (func not in functionOneInpSet) and (func not in functionTwoInpSet) and (func not in variablesSet) and (func not in constsSet):
			raise AssertionError("found an undefined category", func)
		root = EquationTree(func=func, args=[])
		key = children[0]
		if key == 'Half':
			root.varname = '1/2'
		elif key in variables:
			root.varname = 'var_%d' %(variables[key])
		elif negOneFlag:
			root.varname = str(-1)
		else:
			root.varname = str(key)

		return root

	if func == 'Compare' and children[1] == '==':
		func = 'Equality'
		root = EquationTree(func=func, args =[buildAxiom(children[0], variables), buildAxiom(children[2], variables)])
	elif func == 'Compare' and children[1] != '==':
		func = 'Equality'
		raise AssertionError('comparison other than equality is not currently supported')
	else:
		root = EquationTree(func=func, args =[buildAxiom(arg, variables) for arg in children])

	if (func not in functionOneInpSet) and (func not in functionTwoInpSet) and (func not in variablesSet) and (func not in constsSet):
		raise AssertionError("found an undefined category", func)

	return root

# TODO: take this inside matchSubTree:
def matchLeaves(eqTree, subTree):
	# boolean function
	# indicates if two leaves match
	# a leaf can only match a leaf
	eqTreeCopy = copy.deepcopy(eqTree)
	subTreeCopy = copy.deepcopy(subTree)
	if subTreeCopy.isLeaf() == False:
		return False
	elif eqTreeCopy.func in variablesSet:
		# TODO: if the equation is a variable we can replace it with any leaf node as long as we 
		# replace all the instances of that variable with the same symbol and then do the new node 
		# replacement in the tree
		# right now it only matches to a symbol
		if subTreeCopy.func == eqTreeCopy.func:
			return True
		else:
			return False
	elif eqTreeCopy.func in constsSet:
		if (subTreeCopy.func == eqTreeCopy.func) and (subTreeCopy.varname == eqTreeCopy.varname):
			return True
		else:
			return False
	else:
		# print "eqTreeCopy functionality: ", eqTreeCopy.func
		# print "subTreeCopy functionality: ", subTreeCopy.func
		raise AssertionError("we cannot end up here")

def mapLeaveVars(eqTree, subTree):
	lMapping={}
	eqTreeCopy = copy.deepcopy(eqTree)
	subTreeCopy = copy.deepcopy(subTree)
	# Note if the symbol on the rhs and lhs of the equation
	# were non-matching this will not work and we might want to do the varname 
	# lookup
	if eqTreeCopy.func in variablesSet:
		lMapping[subTreeCopy] = copy.deepcopy(eqTreeCopy)
	# print "lMapping:"
	# for k, v in lMapping.iteritems():
	# 	print "k:", k, ": v:", v
	return lMapping

# TODO: take this inside matchSubTree:
def matchTrees(eqTree, subTree):
	# boolean function
	# indicate if subTree is a sub tree of the equation tree
	
	# TODO: if there are several instances of the same variable in the subTree,
	# it should make sure that the eqTree also have these shared
	# simple test case for this situation is the RHS of:
	# (x + y) * z == (x * z) + (y * z)
	eqTreeCopy = copy.deepcopy(eqTree)
	subTreeCopy = copy.deepcopy(subTree)
	if subTreeCopy.isLeaf():
		if subTreeCopy.func in variablesSet:
			# be careful with this
			return True
		elif subTreeCopy.func in constsSet:
			if subTreeCopy.func == eqTreeCopy.func and subTreeCopy.varname == eqTreeCopy.varname:
				# print "subTreeCopy True: ", subTreeCopy.varname, "eqTreeCopy True: ", eqTreeCopy.varname
				return True
			elif subTreeCopy.func == eqTreeCopy.func:
				# raise AssertionError(
				# 	"something weird is going on, constants have the same functionality but",
				# 	"different value. This could be because we have an integer. check ")
				# print "subTreeCopy False: ", subTreeCopy.varname, "eqTreeCopy False: ", eqTreeCopy.varname
				return False
			else: return False
		else:
			raise AssertionError("undefined functionality in the leaf:", subTreeCopy.func)
		# print len(eqTreeCopy.args)
		# print subTreeCopy.func 
		# print eqTreeCopy.func
	if eqTreeCopy.func == subTreeCopy.func:
		match = True
		if len(eqTreeCopy.args) == len(subTreeCopy.args):
			for i in range(0, len(eqTreeCopy.args)):
				match = match and matchTrees(eqTreeCopy.args[i], subTreeCopy.args[i])
		else: match = False
	else:
		match = False
	return match

def mapTreeVars(eqTree, subTree, tMapping):
	# Note if the symbol on the rhs and lhs of the equation
	# were non-matching this will not work and we might want to do the varname 
	# lookup
	eqTreeCopy = copy.deepcopy(eqTree)
	subTreeCopy = copy.deepcopy(subTree)
	if len(subTreeCopy.args)==0:
		tMapping[subTreeCopy] = eqTreeCopy
		# tMapping[subTree.varname] = copy.deepcopy(eqTree)
		return tMapping

	assert len(eqTreeCopy.args)==len(subTreeCopy.args), "reported match is incorrect"
	for i in range(0, len(subTreeCopy.args)):
		tMapping = mapTreeVars(eqTreeCopy.args[i], subTreeCopy.args[i], tMapping)

	# print "tMapping:"
	# for k, v in tMapping.iteritems():
	# 	print "k:", k, ": v:", v
	return tMapping


def matchSubTree(eqTree, mathDictionary):
	# Call on subtree to find a match in the dictionary
	eqTreeCopy = copy.deepcopy(eqTree)
	matches = {}
	varMap = {}
	leafFlag = eqTreeCopy.isLeaf()
	for key, value in mathDictionary.iteritems():
		if leafFlag:
			if matchLeaves(eqTreeCopy, key):
				matches[key] = copy.deepcopy(value)
				varMap[key] = copy.deepcopy(mapLeaveVars(eqTreeCopy, key))
		else:
			if eqTreeCopy.func != key.func:
				continue
			elif matchTrees(eqTreeCopy, key): 
				matches[key] = copy.deepcopy(value)
				varMap[key] = copy.deepcopy(mapTreeVars(eqTreeCopy, key, tMapping={}))

	return matches, varMap

def putNodeInTree(eqTree, newNode, nodeNum):
	newNodeCopy = copy.deepcopy(newNode)
	if len(eqTree.args) == 0:
		#base case: leaf
		if eqTree.number == nodeNum:
			newRoot = newNodeCopy
		else:
			newRoot = copy.deepcopy(eqTree)

	else:
		newRoot = copy.deepcopy(eqTree)
		newRoot.args = []
		for arg in eqTree.args:
			if arg.number==nodeNum:
				newRoot.args.append(newNodeCopy)
			else:
				newRoot.args.append(putNodeInTree(arg, newNodeCopy, nodeNum))
				
	return newRoot

def putSubtreeInTree(eqTree, node, pattern, sub, nodeNum, mapping):
	# This code replaces the 'eqTree's 'node' numbered 'nodeNum', whose pattern is 'pattern',
	# with 'sub'. 'mapping' is the mapping from the sub variables to the eqTree variables
	subCCopy = copy.deepcopy(sub)
	def correctSub(sub, mapping):
		sebCopy = copy.deepcopy(sub)
		if len(sebCopy.args) == 0:
			# print "sebCopy:", sebCopy, isinstance(sebCopy, EquationTree)
			# print "mapping:", mapping
			for k, v in mapping.iteritems():
				if sebCopy == k:
					return copy.deepcopy(mapping[k])
					# print "var:", k, isinstance(k, EquationTree)
					# print "are they equal?:", sebCopy == k
			return sebCopy
		else:
			newSub = copy.deepcopy(sebCopy)
			newSub.args = []
			for arg in sebCopy.args:
				newSub.args.append(correctSub(arg, mapping))
		return newSub	

	if mapping == {}:
		newSub = subCCopy
	else:
		newSub = correctSub(subCCopy, mapping)
	newRoot = putNodeInTree(eqTree, newSub, nodeNum)

	return newRoot


def genPosEquation(eqTree, mathDictionary, eqVars):
	eq = copy.deepcopy(eqTree)
	#randomly choose a node to change:

	# print "equation:", eq
	# print "number of nodes:", eq.numNodes
	nodeNum = random.randrange(0, eq.numNodes)
	dummy = EquationTree(args=[eq])
	# alter the node:
	node = dummy.findNode(nodeNum)

	if node == None:
		print "node number to look for:", nodeNum
		print "node is none. Traversing the equation:"
		eq.preOrder()
		print "also traversing the original equation:"
		eqTree.preOrder()
		raise AssertionError, "node should not be none. There is a problem with numbering"

	# print "node functionality: ", node.func
	# print "equation:", eqTree
	if nodeNum == 0:
		"grow sides"
		newEq = node.growBothSides(eqVars)
		if not isinstance(newEq, EquationTree):
			raise AssertionError("newNode is not of type EquationTree")
	else:
		# print "found node:", node
		# print "nodeNumber:", nodeNum
		matches, varMatchMap = matchSubTree(node, mathDictionary)
		while matches == {}:
			nodeNum = random.randrange(0, eq.numNodes)
			node = dummy.findNode(nodeNum)

			if node == None:
				print "In inner loop: node number to look for:", nodeNum
				print "node is none. Traversing the equation:"
				eq.preOrder()
				print "also traversing the original equation:"
				eqTree.preOrder()
				raise AssertionError, "node should not be none. There is a problem with numbering"


			matches, varMatchMap = matchSubTree(node, mathDictionary)


		# for key, value in matches.iteritems():
		# 	vard = varMatchMap[key]
		# 	print "key:", key, ": value :", value
		# 	print "varKey:", key
		# 	for k, v in vard.iteritems():
		# 		print "k", k, ": v", v

		# print "another printing method just to check:"
		# for key, value in varMatchMap.iteritems():
		# 	print "key:", key
		# 	print "val:", value

		# choose a match randomly:
		match, sub = random.choice(list(matches.items()))
		varMapFinal = varMatchMap[match]
		# print "match:", match
		# print "sub:", sub
		# for k, v in varMapFinal.iteritems():
		# 	print "key:", k, "value:", v

		# replace the match into the tree
		newEq = putSubtreeInTree(eq, node, match, sub, nodeNum, varMapFinal)

	return newEq

def genNegEquation(eqTree, eqTreeVariables):
	global recDepth
	eq = copy.deepcopy(eqTree)
	#randomly choose a node to change:
	nodeNum = random.randrange(1, eq.numNodes) # we don't want to choose the root

	# find the node:
	dummy = EquationTree(args=[eq])
	# alter the node:
	node = dummy.findNode(nodeNum)
	if node == None:
		print "node number to look for:", nodeNum
		print "node is none. Traversing the equation:"
		eq.preOrder()
		raise AssertionError, "node should not be none. There is a problem with numbering"
	# if nodeNum == 0:
	# 	newEq = node.growBothSides()
	# 	if not isinstance(newEq, EquationTree):
	# 		raise AssertionError("newNode is not of type EquationTree")
	# else:
	recDepth = count()
	newNode = node.changeNode(eqTreeVariables)
	# traverse and replace the new node:
	newEq = putNodeInTree(eq, newNode, nodeNum)
	if not isinstance(newNode, EquationTree):
		# print newNode
		raise AssertionError("newNode is not of type EquationTree")
	# after insertion of the new nodes numberings have changed
	globarCtr = count()
	newEq.enumerize()
	return newEq

def isCorrect(eqTree):
	sympyEq = eqTree.pretty()
	print eqTree.prettyOld()
	if sympify(sympyEq) == True:
		return True
	else:
		return False

################################################################################
# NN functions: # 
def buildNNTree(treeType, parsedEquation, num_hidden, params, emb_dimension, varDict={}):
	# TODO: handle range
	if not isinstance(parsedEquation, EquationTree):
		raise AssertionError("input equation should be of type EquationTree")
	if parsedEquation.number == -1:
		raise AssertionError("enumerize the input equation before feeding to buildNNTree")

	func = parsedEquation.func

	if func in variables:
		if parsedEquation.varname == '':
			raise AssertionError("leaf node %s deos not have variable name:" %(str(parsedEquation)))
		root = treeType(prefix='Variable', num_hidden=num_hidden, 
			            params=params, inputName=parsedEquation.varname, args=[],
			            emb_dimension=1, nodeNumber=parsedEquation.number)
						#emb_dimension=len(functionDictionary)
	elif func in consts:
		if parsedEquation.varname == '':
			raise AssertionError("leaf node %s deos not have variable name:" %(str(parsedEquation)))
		root = treeType(prefix='Const', num_hidden=num_hidden,
		                params=params, inputName=parsedEquation.varname, args=[], 
		                emb_dimension=1, nodeNumber=parsedEquation.number)
						#emb_dimension=len(functionDictionary)
	elif func in functionVocab:
		if parsedEquation.varname!='':
			raise AssertionError("non leaf node has input")

		root = treeType(prefix=func, num_hidden=num_hidden, 
			            params=params, args=[], inputName='',
			            emb_dimension=emb_dimension, nodeNumber=parsedEquation.number)
						#emb_dimension=len(functionDictionary)
	else:
		raise ValueError('unknown function %s! add to function list' %(func))

	for arg in parsedEquation.args:
		root.args.append(buildNNTree(treeType=treeType, parsedEquation=arg, 
			                       num_hidden=num_hidden, params=root._params, 
			                       emb_dimension=emb_dimension))

	return root

def get_indices(eq, code=[]):
	eqStr = eq.func
	
	if len(eq.args)==0:
		# if functionDictionary[eqStr] not in set(code):
		code.append(copy.deepcopy(functionDictionary[eqStr]))
	
	for arg in eq.args:
		get_indices(arg, code)

	return code

def encode_equations(equations, vocab, invalid_label=-1, invalid_key='\n', start_label=0):

	idx = start_label
	if vocab is None:
		vocab = {invalid_key: invalid_label}
		new_vocab = True
	else:
		new_vocab = False
	res = []
	for eq in equations:
		# coded = []
		coded = get_indices(eq, [])
		# print "coded:", coded
		res.append(coded)	

	return res, vocab

################################################################################
# classes: #
class EquationTree(object):

	def __init__(self, func='', args=[], varname='', number=-1):
		self.func = func
		self.args = args
		self.varname = varname
		self.number = number

	def __str__(self):
		return self.pretty()

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			# If two trees have the same traversal, would they necassarily be the same?
			# I guess not necessarily, but this serves our purpose for now
			# Might need to add a better equality measure later
			return self.prettyOld()==other.prettyOld()
		else:
			return False

	def __ne__(self, other):
		return not self.__eq__(other)

	def prettyOld(self):
		#pre order traverse
		if len(self.args)!=0:
			tmpStr = self.func + '('
		else:
			tmpStr = self.func + '(' + (self.varname) + ')'
			# if isinstance(self.varname,int):
			# 	name = str(self.varname)
			# else:
			# 	name = self.varname
			# tmpStr = self.func + '(' + name + ')'

		for arg in self.args:
			# print arg.func
			tmpStr = tmpStr + arg.prettyOld() + ','

		if len(self.args)!=0:
			tmpStr = tmpStr[:-1]
			tmpStr = tmpStr + ')'

		return tmpStr

	def pretty(self):
		if len(self.args)!=0:
			if self.func == 'Equality':
				tmpStr = 'Eq' + '('
			else:
				tmpStr = self.func + '('
		else:
			if self.varname == '':
				tmpStr = self.func
			else:
				tmpStr = self.varname

		for arg in self.args:
			# print arg.func
			tmpStr = tmpStr + arg.pretty() + ','

		if len(self.args)!=0:
			tmpStr = tmpStr[:-1]
			tmpStr = tmpStr + ')'

		return tmpStr


	def isLeaf(self):
		if len(self.args) == 0:
			return True
		else:
			return False

	def preOrder(self):
		# if self.func == 'NegativeOne':
		# 	print "negativeOne variable:", self.varname, isinstance(self.varname, str)
		if len(self.args)==0:
			print "variable name:", self.varname
		print "functionality:", self.func
		print "assigned node number:", self.number
		if len(self.args) > 2:
			raise AssertionError('not a binary tree! check buildEq')
		for arg in self.args:
			arg.preOrder()

	def enumerize(self):
		global globalCtr
		def reset(root):
			if root.number != -1:
				root.number = -1
			for arg in root.args:
				reset(arg)

		def enumerizeRecurse(root):
			if root.number == -1:
				root.number = next(globalCtr)

			for i, arg in enumerate(root.args):
				enumerizeRecurse(arg)

		if self.number != -1:
			reset(self)
		# print "reset traversal: beg"
		# self.preOrder()
		# print "reset traversal: end"
		enumerizeRecurse(self)
		# self.preOrder()
		# print "enumerization traversal: end"
		# counting starts from zero, so:
		self.numNodes = next(globalCtr)

	def findNode(self, nodeNum):
		# if the node is a leaf we should change it in both sides, OW it is an incorrect (but valid) eq

		if len(self.args) == 0:
			#base case: leaf
			if self.number == nodeNum:
				# print "arg found. leaf", "self:", self.func, "argnumber:", self.number
				newArg = copy.deepcopy(self)
				return newArg
			else:
				newArg = None
		else:
			for arg in self.args:
				if arg.number==nodeNum:
					# print "arg found. parent:", self.func, "self:", arg.func, "argnumber:", arg.number
					newArg = copy.deepcopy(arg)
					# print newArg
					return newArg
				else:
					newArg = arg.findNode(nodeNum)
					if newArg != None:
						return newArg
		return newArg

	def growBothSides(self, eqVars):
		# TODO
		selfCopy = copy.deepcopy(self)
		randFunc = random.choice(functionBothSides)
		if randFunc in functionTwoInpSet:
			# print randFunc
			if randFunc == 'log':
				orderFlag = 0
				randInpBlock = consts
			else:
				orderFlag = random.choice([0,1])
				randInpBlock = random.choice([consts, variables])

			if randInpBlock == consts:
				randInp = random.choice(randInpBlock)
				if randInp == 'Integer':
					randInt = str(random.choice(intList))
				elif randInp == 'Rational':
					randInt = str(random.choice(ratList))
				elif randInp == 'Float':
					randInt = str(random.choice(floatList))
				elif randInp == 'One':
					randInt = str(1)
				elif randInp == 'NegativeOne':
					randInt = str(-1)
				elif randInp == 'Pi':
					randInt = 'pi'
				elif randInp == 'Half':
					randInt = str(1/2)
				else:
					randInt = randInp
			else:
				randInp  = 'Symbol'
				if eqVars == {}:
					randInt = 'var_0'
					eqVars[randInt] = 0
				else:
					varKeys = [eqVars[k] for k in eqVars.keys()]
					varKeys.sort()
					# print "eqVars:", eqVars
					# print "varKeys:", varKeys
					randInt = 'var_%s'%(str(varKeys[-1]+1))
					# randInt = 'var_%s'%(str(len(eqVars)))
					if randInt in eqVars:
						print "variable dictionary:", eqVars
						print "randInt:", randInt	
						raise AssertionError('wrongly calculated dictionary size')

					eqVars[randInt] = varKeys[-1]+1

			# print randInp
			inp = EquationTree(func=randInp, args=[], varname=randInt)
			selfCopy.args = []
			if orderFlag == 0:
				selfCopy.args.append(EquationTree(func=randFunc, args=[self.args[0], inp]))
				selfCopy.args.append(EquationTree(func=randFunc, args=[self.args[1], inp]))
			else:
				selfCopy.args.append(EquationTree(func=randFunc, args=[inp, self.args[0]]))
				selfCopy.args.append(EquationTree(func=randFunc, args=[inp, self.args[1]]))
		elif randFunc in functionOneInpSet:
			selfCopy.args = []
			selfCopy.args.append(EquationTree(func=randFunc, args=[self.args[0]]))
			selfCopy.args.append(EquationTree(func=randFunc, args=[self.args[1]]))
		else:
			raise AssertionError("equality should not end up here")

		return selfCopy

	def growNode(self, eqVars):
		# we might be able to remove the deepCopies since the input itself is deep copied from orig
		selfCopy = copy.deepcopy(self)
		possibleFuncs = [functionOneInp, functionTwoInp]
		# print "possibleFuncs:", possibleFuncs
		randChoice = random.choice(possibleFuncs)
		# print "randChoice:", randChoice
		randFunc = random.choice(randChoice)
		# print "randFunc:", randFunc
		if randChoice == functionOneInp:
			# print "in functionOneInp"
			newSelf = EquationTree(func=randFunc, args=[selfCopy])
			# print newSelf.args
			return newSelf
		else:
			# print "in else"
			possibleInputs = [variables, consts]
			randInp = random.choice(possibleInputs)
			inputName = random.choice(randInp)
			# print "inputName:", inputName
			if randInp == variables:
				# choose the variable name from the possible variables in the equation
				# print "eqVars:", eqVars
				if eqVars != {}:
					randVar = random.choice(eqVars.keys())
				else:
					randVar = 'var_0'
					eqVars[randVar] = 0
				# print "randVar:", randVar
				inp = EquationTree(func=inputName, args=[], varname='var_%s'%(str(eqVars[randVar])))
			else: 
				if inputName == 'NegativeOne':
					randVar = str(-1)
				elif inputName == 'One':
					randVar = str(1)
				elif inputName == 'Integer':
					randVar = str(random.choice(intList))
				elif inputName == 'Rational':
					randVar = str(random.choice(ratList))
				elif inputName == 'Float':
					randVar = str(random.choice(floatList))
				elif inputName == 'Half':
					randVar = str(1/2)
				elif inputName == 'Pi':
					randVar = 'pi'
				else:
					randVar = inputName

				if randVar == '':
					raise AssertionError("leaf node %s should have input name" %(inputName))

				# randVar = random.choice(eqVars.keys())
				inp = EquationTree(func=inputName, args=[], varname=randVar)

			newSelf = EquationTree(func=randFunc, args=[selfCopy, inp])
			return newSelf


	def shrinkNode(self, eqVars):
		# we might be able to remove the deepCopies since the input itself is deep copied from orig
		selfCopy = copy.deepcopy(self)
		if len(selfCopy.args) == 0:
			#if it's a leaf we can't shrink it further
			return selfCopy
		chosenNode = random.choice(selfCopy.args)
		selfCopy = chosenNode
		return selfCopy


	def replaceNode(self, eqVars):
		# this will almost always give an incorrect equation
		# we might be able to remove the deepCopies since the input itself is deep copied from orig
		selfCopy = copy.deepcopy(self)
		# print selfCopy

		assert len(selfCopy.args) <= 2, "tree is not binary!"

		if len(selfCopy.args) == 0:
			# print "leaf"
			if selfCopy.func in variablesSet:
				# is it is a Symbol, better not replace it since we should maintain the same symbol in both sides of the equality
				# print "in variable sets"
				if len(eqVars) == 0:
					raise AssertionError("variable %s is not listed in eqVars %s" %(selfCopy.func, str(eqVars)))
				elif len(eqVars) == 1:
					for key in eqVars.keys():
						randVar = eqVars[key] + 1
					eqVars['var_%s'%(randVar)] = randVar
					selfCopy.varname = 'var_%s'%(randVar)
				else:
					wctr = 0
					randVar = random.choice(eqVars.keys())
					while 'var_%s'%(eqVars[randVar]) == selfCopy.varname:
						wctr += wctr
						if wctr > 10:
							raise AssertionError('not able to choose a valid variable name among %s' %(str(eqVars)))
						randVar = random.choice(eqVars.keys())

					selfCopy.varname = 'var_%s'%(eqVars[randVar])
				return selfCopy

			elif selfCopy.func in constsSet:
				# print "in constant sets"
				randFunc = random.choice(consts)
				while randFunc == selfCopy.func:
					randFunc = random.choice(consts)
				
				if randFunc == 'NegativeOne':
					randVar = str(-1)
				elif randFunc == 'One':
					randVar = str(1)
				elif randFunc == 'Integer':
					randVar = str(random.choice(intList))
				elif randFunc == 'Rational':
					randVar = str(random.choice(ratList))
				elif randFunc == 'Float':
					randVar = str(random.choice(floatList))
				elif randFunc == 'Half':
					randVar = str(1/2)
				elif randFunc == 'Pi':
					randVar = 'pi'
				else:
					randVar = randFunc

				if randVar == '':
					raise AssertionError("leaf node %s should have input name" %(randFunc))

				return EquationTree(func=randFunc, args=[], varname=randVar)
			else:
				raise AssertionError("leaf functionality %s is not in the variables or conts" %(selfCopy.func))
				# return selfCopy

		elif len(selfCopy.args) == 1:
			# print "one input"
			# TODO: we can replace a sin(arg) with cos(pi/2 - arg) to produce correct equations
			# TODO: what happens to the rest of them?
			randFunc = random.choice(functionOneInp)
			while randFunc == selfCopy.func:
				randFunc = random.choice(functionOneInp)
			return EquationTree(func=randFunc, args=selfCopy.args)

		elif len(selfCopy.args) == 2:
			# print "two inputs"
			randFunc = random.choice(functionTwoInp)
			while randFunc == selfCopy.func:
				randFunc = random.choice(functionTwoInp)
			return EquationTree(func=randFunc, args=selfCopy.args)

	def changeNode(self, eqVars):
		global recDepth
		global maxDepth
		selfCopy = copy.deepcopy(self)
		# TODO: I think I should do these for a certain number of times to make more changes.
		# Currently this changes only one node
		methods = [selfCopy.growNode, selfCopy.shrinkNode, selfCopy.replaceNode]
		# This makes each method equally likely. Another method like Sameer's code is to generate
		# all candidate nodes and then randomly choose one of them 
		method = random.choice(methods)
		newSelf = method(eqVars)
		# newSelf = self.growNode(eqVars)
		if newSelf == selfCopy and next(recDepth) < maxDepth:
			# print "they are equal"
			selfCopy.changeNode(eqVars)
		# if newSelf == None:
		# print method
		return newSelf

	def extractVars(self):
		exVars = {}
		def extractVarsRecurse(root):
			if len(root.args) == 0 and root.func in variablesSet:
				if root.varname not in exVars:
					# print "root.varname", root.varname
					exVars[root.varname] = int(re.split('_', root.varname)[-1])
			for arg in root.args:
				extractVarsRecurse(arg)

		extractVarsRecurse(self)
		return exVars


################################################################################
# main: #

def main():
	inputPath = "eqTest.json"
	axiomPath = "smallAxioms.txt"
	jsonAtts = ["equation", "range", "variables","labels"]

	import logging
	head = '%(asctime)-15s %(message)s'
	logging.basicConfig(level=logging.DEBUG, format=head)

	## equation atts:
	labels = []
	inputRanges = []
	inputEquations = [] 
	eqVariables = []
	parsedEquations = []
	parsedRanges = []
	equations = []
	ranges = []
	inputAxioms = []
	axiomVariables = []
	axioms = []
	axLabels = []

	global recDepth
	global globalCtr
	global maxDepth
	maxDepth = 10

	numPosEq = 5
	numNegEq = 5

	## NN atts:
	params = None
	contexts = mx.cpu(0)
	num_hidden = 100
	vocabSize = len(functionDictionary)
	emb_dimension = 16
	out_dimension = 32
	batch_size = 1
	split = 0.8


	#reading input and parsing input equations
	readJson(inputPath, inputEquations, inputRanges, jsonAtts)
	readAxioms(axiomPath, inputAxioms, axiomVariables)
	parseEquation(inputEquations, parsedEquations, eqVariables)
	parseRange(inputRanges, parsedRanges, eqVariables)
	print "ranges:", inputRanges
	print "parsedRanges:", parsedRanges
	assert 1==2
	# print "equation zero:", parsedEquations[0]
	for i, eq in enumerate(parsedEquations):
		currEq = buildEq(eq, eqVariables[i])
		globalCtr = count()
		currEq.enumerize()
		# print "currEq:", currEq
		# currEq.preOrder()
		equations.append(currEq)

		# the first equation in the input function is incorrect. It has been deliberately added
		# to include all possible functionalities in the functionDictionary. 
		# This is becuase of mxnet's inability to add extra parameters to the parameter list as new equations
		# are loaded
		if i == 0:
			labels.append(mx.nd.array([0]))
		else:
			labels.append(mx.nd.array([1]))


	# print "labels[0]", labels[0].asnumpy()
	# a = labels[0].asnumpy()==np.array([0])
	# print "isZero:", a[0]
	# print equations[1]
	# p = preOrder(equations[1])
	# print [p[i][:-1] for i in range(len(p))]
	# assert 1==2

	numSamples = len(parsedEquations)
	buckets = list(xrange(numSamples))

	# print isCorrect(equations[10])
	# assert 1==2, 'stop'

	# print "equation zero:", equations[0]
	# equations[0].preOrder()

	for i, ax in enumerate(inputAxioms):
		currAxiom = buildAxiom(ax, axiomVariables[i])
		globalCtr = count()
		currAxiom.enumerize()
		# print "currAxiom", currAxiom
		# currAxiom.preOrder()
		axioms.append(currAxiom)
		axLabels.append(mx.nd.array([1]))
		
	equations.extend(axioms)
	eqVariables.extend(axiomVariables)
	labels.extend(axLabels)

	# print axioms[8]
	# axioms[8].preOrder()

	# for eq in equations:
	# 	print eq
	# 	eq.preOrder()
	# 	print "number of nodes:", eq.numNodes

	# assert 1==2, 'stop'

	mathDictionary = {}
	for eq in equations:
		eqCopy = copy.deepcopy(eq)
		# eqCopy.preOrder()
		if eqCopy.args[0] not in mathDictionary:
			mathDictionary[eqCopy.args[0]] = eqCopy.args[1]
		if eqCopy.args[1] not in mathDictionary:
			mathDictionary[eqCopy.args[1]] = eqCopy.args[0]
	# for key, value in mathDictionary.iteritems():
	# 	print "key: ", key
	# 	print "value: ", value

	# posEquations = []
	# posLabels = []
	for i in range(0, numPosEq):
		randInd = random.choice(range(len(equations)))
		# print "randInd:", randInd
		while labels[randInd].asnumpy() == np.array([0]):# labels[0].asnumpy()==np.array([0])
			randInd = random.choice(range(len(equations)))
		randEq = equations[randInd]
		randEqVariable = copy.deepcopy(eqVariables[randInd])
		# print "posEqVars:", randEqVariable

		posEq = genPosEquation(randEq, mathDictionary, randEqVariable)
		posVars = posEq.extractVars()
		globalCtr = count()
		posEq.enumerize()
		while posEq in mathDictionary:
			print "repeted pos equation"
			posEq = genPosEquation(randEq, mathDictionary, randEqVariable)
			posVars = posEq.extractVars()
			globalCtr = count()
			posEq.enumerize()
		# print "posEq:", posEq
		# posEq.preOrder()

		posEqCopy = copy.deepcopy(posEq)
		if posEqCopy.args[0] not in mathDictionary:
			mathDictionary[posEqCopy.args[0]] = posEqCopy.args[1]
		if posEqCopy.args[1] not in mathDictionary:
			mathDictionary[posEqCopy.args[1]] = posEqCopy.args[0]
		
		equations.append(posEq)
		eqVariables.append(posVars)
		labels.append(mx.nd.array([1]))

	# assert 1==2
	# equations.extend(posEquations)
	# labels.extend(posLabels)

	negLabels= []
	negEquations = []
	negEqVariables = []
	negMathDictionary = {}
	for i in range(0, numNegEq):
		randInd = random.choice(range(len(equations)))
		# print "randInd:", randInd
		randEq = equations[randInd]
		# print "negRandEq:", randEq
		randEqVariable = copy.deepcopy(eqVariables[randInd])
		# print "negRandEqVariable:", randEqVariable
		# print "randEq:", randEq
		# print "negative equation traversal:"
		# randEq.preOrder()
		negEq = genNegEquation(randEq, randEqVariable)
		negVars = negEq.extractVars()
		globalCtr = count()
		negEq.enumerize()
		while negEq in negMathDictionary:
			print "repeated neg equation"
			negEq = genNegEquation(randEq, randEqVariable)
			negVars = negEq.extractVars()
			globalCtr = count()
			negEq.enumerize()

		negEqCopy = copy.deepcopy(negEq)
		if negEqCopy.args[0] not in negMathDictionary:
			negMathDictionary[negEqCopy.args[0]] = negEqCopy.args[1]
		if negEqCopy.args[1] not in negMathDictionary:
			negMathDictionary[negEqCopy.args[1]] = negEqCopy.args[0]

		# print "negEq:", negEq
		# negEq.preOrder()
		negEquations.append(negEq)
		negEqVariables.append(negVars)
		# if isCorrect(negEq):
		# 	negLabels.append(mx.nd.array([1]))	
		# else:
		negLabels.append(mx.nd.array([0]))


	# assert 1==2, "stop"

	# print "**************************************psitive equations**************************************"
	# for eq in equations:
	# 	print "pos equation:", eq
	# 	print "is correct:", isCorrect(eq)
	# print "**************************************negative equations**************************************"
	# for eq in negEquations:
	# 	print "neg equation:", eq
	# 	print "is correct:", isCorrect(eq)

	equations.extend(negEquations)
	labels.extend(negLabels)
	eqVariables.extend(negEqVariables)

	# writeJson("data.json", equations, ranges, eqVariables, labels)
	# a = preOrder(equations[18])
	# # print a
	# currEq = makeEqTree(a[0].split(','), a[1].split(','), a[2].split(','), equations[18].numNodes)
	# print "old equation:"
	# equations[18].preOrder()
	# print "new equation:"
	# currEq.preOrder()
	# assert 1==2

	# pprint.pprint([equations[i].prettyOld() for i in range(len(equations))])
	# assert 1==2

		# TODO: write a function that generates sympy understandable equations to check correctness for label

		# this is to check how good sympy is in finding correct equations. It returns some as false, because pi is defined as a
		# free_symbol in stead of pi, which is caused by process_sympy
		# I also tried inputing sympy expressions, but the evaluate = False flag sometimes does not work, so it evaluates the expression
		# to true in stead of parsing it
		# These also indicate that the equations are parsed incorrectly. I should find a hack to convert symbol('pi') back to pi
		# if simplify(eq)==True :# or (trigsimp(eq)==True): #trigsimp gives some funny errors
		# CORRECTION: The above is not really an issue. We know the input is correct, when we generate a new expression, we can 
		# control the resulting equation and make sure that the final equation is correct
		# 	labels.append(1) #input equations are positive
		# else:
		# 	labels.append(0)
			# print simplify(eq)
			# print eq
			# raise AssertionError('input equation %d is incorrect', i)

	# for eq in newEquations:
	# 	print eq
		# eq.preOrder()

	samples = []
	dataNames = []
	for i, eq in enumerate(equations):
		# print "label:", labels[i].asnumpy()
		# print "equation:", eq
		currNNTree = buildNNTree(treeType=nnTree , parsedEquation=eq, 
			                         num_hidden=num_hidden, params=params, 
			                         emb_dimension=emb_dimension)

		# print currNNTree
		currDataNames = currNNTree.getDataNames(dataNames=[])
		# print "currDataNames:", currDataNames
		dataNames.append(currDataNames)
		samples.append(currNNTree)

	indices = range(len(equations))
	random.shuffle(indices)
	splitInd = int(split * len(equations))
	trInd = indices[:splitInd]
	teInd = indices[splitInd:]
	trainEquations = [equations[i] for i in trInd]
	testEquations = [equations[i] for i in teInd]
	trainLabels = [labels[i] for i in trInd]
	testLabels = [labels[i] for i in teInd]
	trainVars = [eqVariables for i in trInd]
	testVars = [eqVariables for i in teInd]


	# train_eq, _ = encode_equations(equations, vocab=functionDictionary, invalid_label=-1, invalid_key='\n', start_label=0)
	train_eq, _ = encode_equations(trainEquations, vocab=functionDictionary, invalid_label=-1, invalid_key='\n', start_label=0)
	test_eq, _ = encode_equations(testEquations, vocab=functionDictionary, invalid_label=-1, invalid_key='\n', start_label=0)
	# data_train = mx.rnn.BucketSentenceIter(train_eq, batch_size)
	# data_train  = BucketEqIterator(enEquations=train_eq, eqTreeList=samples, batch_size=batch_size, buckets=buckets, labels=labels,
 #                                            invalid_label=-1)
	data_train  = BucketEqIterator(enEquations=train_eq, eqTreeList=samples, batch_size=batch_size, buckets=buckets, labels=trainLabels,
                                            invalid_label=-1)
	data_test  = BucketEqIterator(enEquations=test_eq, eqTreeList=samples, batch_size=batch_size, buckets=buckets, labels=testLabels,
                                            invalid_label=-1)

	def sym_gen(bucketIDX):
		# print "in sym_gen"
		data = mx.sym.Variable('data')
		label = mx.sym.Variable('softmax_label')
		# embed = mx.sym.Embedding(data=data, input_dim=len(functionVocab),
		#                          output_dim=args.num_embed, name='embed')

		
		# We need to figure out how to use the bucketIDX. 
		# I think the original one handles it using the data iterator.
		# We might be able to handle this using mx.rnn.BucketSentenceIter
		tree = samples[bucketIDX]
		# print "bucketIDX:", bucketIDX
		# print "tree:", equations[bucketIDX]
		# print "label:", labels[bucketIDX].asnumpy()
		dataNames = tree.getDataNames(dataNames=[])
		nameDict = {}
		for dn in set(dataNames):
			if dn not in nameDict:
				nameDict[dn] = mx.sym.Variable(dn)
		outputs = tree.unroll(nameDict)
		# data_names = dataNames[bucketIDX]
		# print data_names
		dataNames = list(set(dataNames))
		# data = mx.sym.Group([value for _, value in dataNames.iteritems()])


		# pred = mx.sym.Reshape(outputs, shape=(-1, tree._num_hidden))
		pred = mx.sym.FullyConnected(data=outputs, num_hidden=out_dimension, name='pred')

		label = mx.sym.Reshape(label, shape=(-1,))
		pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

		# return pred, ('data',), ('softmax_label',)
		return pred, (dataNames), ('softmax_label',)

	model = mx.mod.BucketingModule(
		sym_gen             = sym_gen,
		default_bucket_key  = 0,
		context             = contexts)

	model.fit(
        train_data          = data_train,
        eval_data           = data_test,
        eval_metric         = 'acc',#mx.metric.Perplexity(-1),
        kvstore             = 'str',
        optimizer           = 'sgd',
        optimizer_params    = { 'learning_rate': 0.0001,
                                'momentum': 0.0,
                                'wd': 0.00001 },
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch           = 25)
        # batch_end_callback  = mx.callback.Speedometer(1, 1))
	


if __name__ == "__main__":
	main()