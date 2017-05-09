import sys
# path to latex2sympy. Converts a latex equation to sympy format
sys.path.append('/Users/Forough/Documents/bitBucket/math-knowledge-base/Codes/latex2sympy')

import copy
import json
# from process_latex import process_sympy
from sympy import *
import re
import pprint
import mxnet as mx
import numpy as np
# from tagger import readJson
# from prover import parseEquation
from itertools import count
import random
# from equationGenerator import EquationTree

################################################################################
# math vocabulary:
functionVocab = ['Equality', 'Add', 'Mul', 'Pow',
				  'sin', 'cos', 'csc', 'sec', 'tan', 'cot',
				  'asin', 'acos', 'acsc', 'asec', 'atan', 'atan2', 'acot',
				  'sinh', 'cosh', 'csch', 'sech', 'tanh', 'coth',
				  'asinh', 'acosh', 'acsch', 'asech', 'atanh', 'acoth',
				  'log', 'exp',
				  'Min', 'Max', 'root', 'sqrt', 'IdentityFunction',
				  'range', 'const', 'var']
variables = ['Symbol']
consts = ['NegativeOne', 'NaN', 'Infinity', 'Exp1', 'Pi', 'One', 'Half', 'Integer', 'Rational']
# We don't need to generate a separate class for each of the variables or functions, rather:
# constExprs = [ ConstExpr(e) for e in consts]

tmp = []
tmp.extend(functionVocab)
tmp.extend(variables)
tmp.extend(consts) 

functionDictionary = {}
ctr = 1
for f in tmp:
	functionDictionary[f] = ctr
	ctr+=1

# pprint.pprint(functionDictionary)

################################################################################
# functions: #
treeCounter = count()
def buildTree(treeType, parsedEquation, num_hidden, params, emb_dimension, varDict={}):
	# TODO: handle range
	func = str(parsedEquation.func)
	func = func.split('.')[-1]
	while func[-1]=='\'' or func[-1]=='>':
		func = func[:-1]

	# if params != None:
	# 	print "params:", params._params
	# else:
	# 	print "params:", params

	# root computation
	if func in variables:
		# root = treeType(prefix='variables', num_hidden=num_hidden, params=params, inputName=func, args=[], emb_dimension=len(functionDictionary))
		root = treeType(prefix=func, num_hidden=num_hidden, 
			            params=params, inputName=str(func), args=[],
			            emb_dimension=len(functionDictionary), nodeNumber=next(treeCounter))
	elif func in consts:
		# root = treeType(prefix='const', num_hidden=num_hidden, params=params, inputName=func, args=[], emb_dimension=len(functionDictionary))
		root = treeType(prefix=func, num_hidden=num_hidden,
		                params=params, inputName=str(func), args=[], 
		                emb_dimension=len(functionDictionary), nodeNumber=next(treeCounter))
	elif func in functionVocab:
		root = treeType(prefix=func, num_hidden=num_hidden, 
			            params=params, args=[], inputName='',
			            emb_dimension=emb_dimension, nodeNumber=next(treeCounter))
	else:
		raise ValueError('unknown function! add to function list')

	## added this Wed, Apr 19
	# if len(parsedEquation.args) == 0:
	# 	root.args.append(treeType(prefix='data', num_hidden=num_hidden, params=params, inputName='data', args=[], emb_dimension=len(functionDictionary)))
	## up to here

	# print root.func
	#children computation
	for arg in parsedEquation.args:
		# print arg
		root.args.append(buildTree(treeType=treeType, parsedEquation=arg, 
			                       num_hidden=num_hidden, params=root._params, 
			                       emb_dimension=emb_dimension))

	# print "root args:", len(root.args)
	# print "equation args:", len(parsedEquation.args)
	return root

def one_hot(index, depth):
	out = mx.ndarray.zeros(depth)
	out[index-1] = 1
	return out

def get_indices(eq, code=[]):
	eqStr = str(eq.func)
	eqStr = eqStr.split('.')[-1]
	while eqStr[-1]=='\'' or eqStr[-1]=='>':
		eqStr = eqStr[:-1]
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

class nnTree(mx.rnn.BaseRNNCell):

	def __init__(self, num_hidden, emb_dimension, prefix='',  params=None, args=[], inputName='', nodeNumber=-1):
		super(nnTree, self).__init__(prefix='nnTree_'+prefix+'_', params=params)
		self.args = args
		self.func = prefix
		self.num_hidden = num_hidden
		self.emb_dimension = emb_dimension
		self.inputName = inputName
		self.nodeNumber = nodeNumber
		# print "self prefix:", self._prefix
		# print "params prefix", self._params._prefix
		if params is not None:
			self._params._prefix = self._prefix
		self._iW = self._params.get('i2h_weight')
		self._iB = self._params.get('i2h_bias')
		# print "nodeNumber:", nodeNumber

	def __str__(self):
		return self.func

	def __call__(self, inp, children):
		"""Construct symbol for one step of treeRNN.
		Parameters
		----------
		inputs : sym.Variable
		    input symbol, 2D, batch * num_units
		states : sym.Variable
		    state from previous step or begin_state().
		Returns
		-------
		output : Symbol
		    output symbol
		states : Symbol
		    state to next step of RNN.
		"""
		name = '%s_%d_' % (self._prefix, self.nodeNumber)
		#name = '%s_'%(self.prefix) # it was self._prefix before. Why tho? changed it back to self.prefix

		if children!=None and inp!=None:
			raise ValueError("cannot have both an input and children")

		if children==None:
			if not isinstance(inp, mx.symbol.Symbol):
				print "not instance:", inp
				if inp==None:
					raise AssertionError("leaf node %s does not have input" %(str(self)))
				else:
					raise AssertionError("unknown type for input: %s" %(str(type(inp))))
			#leaf
			data = inp
			# print "inferred shape:", data.infer_shape()

		elif children==[]:
			print "self.inputName:", self.inputName
			raise AssertionError("something weird is going on. inputName is %s and func is %s" %(str(inp), str(self)))

		elif inp==None:
			#not leaf
			if len(children) == 0:
				raise AssertionError('child node of %s does not have input' %(str(self)))
			if len(children)==1:
				data = children[0]
			elif len(children)==2:
				# data = mx.symbol.Concat(children[0], children[1])
				data = mx.symbol.concat(children[0], children[1], dim=1)
				# data = children[0] + children[1]
			elif len(children)>2:
				print "parent:", self
				print "children:", [children[i] for i in range(len(children))]
				raise ValueError("the number of children should not exceed 2")

		else: 
			raise AssertionError("nor leaf nor non-leaf!!!")
			
			# i think weight sharing is not complete because _iw is not linked to the selfname.
			# make sure it is: This is handled now
		# print "self weight", self._iW
		# print "network name", '%si2h'%name
		i2h = mx.symbol.FullyConnected(data=data, weight=self._iW, bias=self._iB,
	                            num_hidden=self.num_hidden,
	                            name='%si2h'%name)
		state = mx.symbol.Activation(data=i2h, act_type="sigmoid", name='%sstate'%name)

		return state

	def unroll(self, dataNameDictionary):
		# call unroll on root to compute the output states
		states_children = []
		# ctr+=1
		# print "self function:", self.func
		# print self.args
		# print "arg length: ", len(self.args)

		for arg in self.args:
			states = arg.unroll(dataNameDictionary=dataNameDictionary)
			states_children.append(states)

		if self.inputName=='':
			# print "input from unroll:", self
			output_state = self(inp=None, children=states_children)
		elif self.inputName!='' and len(states_children)!=0:
			raise ValueError("non-leaf node has input!")
		# elif self.inputName = 'data':

		else:
			#in leaf
			# inputIndex = functionDictionary[self.inputName]
			# inputs = mx.symbol.Variable(name=self.inputName)
			if self.inputName == '':
				raise AssertionError, "leaf does not have input name"
			inputs = dataNameDictionary[self.inputName]
			# inputs = mx.symbol.Variable(name='data')
			# we should not do binding here. Binding happens inside the bucketingModule
			# inputs = inputs.bind(mx.cpu(), {self.inputName: one_hot(index=inputIndex, depth=len(functionDictionary))} )
			output_state = self(inp=inputs, children=None)

		return output_state

	def getDataNames(self, dataNames=[]):
		if len(self.args)==0:
			# if self.inputName not in set(dataNames):
			dataNames.append(self.inputName)

		for arg in self.args:
			arg.getDataNames(dataNames)
		# return list(set(dataNames))
		return dataNames

	def traverse(self):
		print self.func
		for arg in self.args:
			arg.traverse()

class BucketEqIterator(mx.io.DataIter):
	"""Simple bucketing iterator for tree LSTM model for equations.
    Label for each step is constructed from data of
    next step.
    Parameters
    ----------
    enEquations : list of list of int
        encoded equations
    batch_size : int
        batch_size of data
    invalid_label : int, default -1
        key for invalid label, e.g. <end-of-sentence>
    dtype : str, default 'float32'
        data type
    buckets : list of int
        size of data buckets. Automatically generated if None.
    data_name : str, default 'data'
        name of data
    label_name : str, default 'softmax_label'
        name of label
    layout : str
        format of data and label. 'NT' means (batch_size, length)
        and 'TN' means (length, batch_size).
    """
	def __init__(self, enEquations, eqTreeList, batch_size, labels, buckets=None, invalid_label=-1,
	             label_name='softmax_label', dtype='float32',
	             layout='NTC'):
		super(BucketEqIterator, self).__init__()

		buckets = np.arange(len(enEquations))
		# print "buckets:", buckets

		self.data = [[] for _ in buckets]
		self.data_name = [[] for _ in buckets]
		# self.data = enEquations # data type should be sat
		for i, eq in enumerate(enEquations):
			buck = i
			# print "eq: ", eq
			eq = list(set(eq)) # TODO: fix this later as well as the datanames
			buff = []
			# for j in range(len(eq)):
			# 	buff.append([np.array(eq[j], dtype=dtype)])
			# buff = [np.array(eq[j], dtype=dtype) for j in range(len(eq))]
			for j in range(len(eq)):
				if isinstance(eq[j],list):
					buff.append(np.array(eq[j], dtype=dtype))
				else:
					buff.append(np.array([eq[j]], dtype=dtype))
			# buff = [np.array([eq[j]], dtype=dtype) for j in range(len(eq))]
			# buff = np.array(eq, dtype=dtype)
			self.data[buck].extend(buff)
			# print "curr tree data:", eqTreeList[i].getDataNames([])

			# print "buff:", buff

			# print "buck:", buck
			# print "eqTreeList:", eqTreeList[0]
			# print "eq", eq
			self.data_name[buck].append(list(set(eqTreeList[i].getDataNames([]))))


		# print "data_name:", self.data_name
		# print "data:", self.data
		# print "data_shape:", len(self.data[0])

		# self.data = [np.asarray(i, dtype=dtype) for i in self.data]

		self.batch_size = batch_size
		self.buckets = buckets
		# self.data_name = data_name
		self.label_name = label_name
		self.dtype = dtype
		self.invalid_label = invalid_label
		self.nddata = []
		self.ndlabel = []
		self.major_axis = layout.find('N')
		self.labels = labels
		# print "self.major_axis:", self.major_axis
		self.default_bucket_key = 0# max(buckets) # what is our default bucket key?

		# print "self.data_name:", self.data_name[self.default_bucket_key][0]
		if self.major_axis == 0:
			# print "here in self.major_axis"
			# self.provide_data = [(self.data_name[self.default_bucket_key][0][i], (batch_size, len(self.data[self.default_bucket_key][0]))) 
			#                       for i in range(len(self.data_name[self.default_bucket_key][0]))]
			self.provide_data = [(self.data_name[self.default_bucket_key][0][i], (batch_size, )) 
			                      for i in range(len(self.data_name[self.default_bucket_key][0]))]
			# self.provide_label = [(label_name, (batch_size, self.default_bucket_key))]
			self.provide_label = [(label_name, (batch_size, ))]
		elif self.major_axis == 1:
			# self.provide_data = [(self.data_name[self.default_bucket_key][0][i], (len(self.data[self.default_bucket_key][0]), batch_size))
			#                       for i in range(len(self.data_name[self.default_bucket_key][0]))]
			self.provide_data = [(self.data_name[self.default_bucket_key][0][i], (1, batch_size))
			                      for i in range(len(self.data_name[self.default_bucket_key][0]))]
			# self.provide_label = [(label_name, (self.default_bucket_key, batch_size))]
			self.provide_label = [(label_name, (1, batch_size))]
		else:
			raise ValueError("Invalid layout %s: Must by NT (batch major) or TN (time major)")


		# print "self.provide_data:", self.provide_data
		# print "self.data:", self.data

		# This is the index of each bucket and the number of equations in each bucket. So for
		# us this is only (i, 0) for i in range(len(self.data))
		self.idx = []
		for i, buck in enumerate(self.data):
			# self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
			self.idx.extend([(i, 0)])
		self.curr_idx = 0

		self.reset()

		# print "self index:", self.idx

		# print "provide_data:", self.provide_data

	def reset(self):
		self.curr_idx = 0
		# random.shuffle(self.idx)
		# for buck in self.data:
		# 	print "buck:", buck
		# 	np.random.shuffle(buck)

		self.nddata = []
		self.ndlabel = []
		for i, buck in enumerate(self.data):
			# print "i in loop:", i
			# print "buck:", buck
			# print "buck len:", len(buck)

			# print "buck[0]:", buck[0]
			# print "buck[1]:", buck[1]
			# print "buck ndarray:", [mx.ndarray.array(buck[k], dtype=self.dtype) for k in range(len(buck))]
			label = self.labels[i]
			# self.nddata.append(mx.ndarray.array(buck, dtype=self.dtype))
			# self.ndlabel.append(mx.ndarray.array(label, dtype=self.dtype))
			self.nddata.append([mx.ndarray.array(buck[k], dtype=self.dtype) for k in range(len(buck))])
			self.ndlabel.append(mx.ndarray.array(label, dtype=self.dtype))
			# self.nddata.append(buck)
			# self.ndlabel.append(mx.ndarray.array(label, dtype=self.dtype))
		# for j in range(len(self.nddata)):
		# 	print "self.nddata",j,":", [self.nddata[j][k].asnumpy() for k in range(len(self.nddata[j]))]
		# # print "self.nddata:", [self.nddata[j] for j in range(len(self.nddata))]
		# print "self.ndlabel:", [self.ndlabel[j].asnumpy() for j in range(len(self.ndlabel))]


	def next(self):
		# print "in next"
		# print "curr index inside next:", self.curr_idx
		if self.curr_idx == len(self.idx):
			raise StopIteration
		i, j = self.idx[self.curr_idx]
		self.curr_idx += 1

		if self.major_axis == 1:
			# data = self.nddata[i][j:j+self.batch_size].T
			# label = self.ndlabel[i][j:j+self.batch_size].T
			data = self.nddata[i].T
			label = self.ndlabel[i].T
		else:
			# data = self.nddata[i][j:j+self.batch_size]
			# label = self.ndlabel[i][j:j+self.batch_size]
			data = self.nddata[i]
			label = self.ndlabel[i]

		# print "i:", i
		# print "j:", j
		# print "data:", data
		# print "label:", label

		# print "data shape:", data.shape
		d = mx.io.DataBatch(data, [label], pad=0,
		                 bucket_key=self.buckets[i],
		                 # provide_data=[(self.data_name[i][0][j], data.shape) for j in range(len(self.data_name[i][0]))],
		                 provide_data=[(self.data_name[i][0][j], (self.batch_size, )) 
		                                for j in range(len(self.data_name[i][0]))],
		                 provide_label=[(self.label_name, label.shape)])

		# print "d.provide_data:", d.provide_data
		# print "d.data:", [d.data[j] for j in range(len(d.data))]
		return d


################################################################################
# main: #
def main():

	# pprint.pprint(functionDictionary)

	# params = mx.rnn.RNNParams()
	params = None
	contexts = mx.cpu(0)
	num_hidden = 100
	vocabSize = len(functionDictionary)
	emb_dimension = 16
	out_dimension = 32
	batch_size = 1

	inputPath = "smallTestMxnet.json"
	jsonAtts = ["variables", "CCGparse", "equation","sentence","equation","equation"]

	parseTreeList = [] # list of lists
	rawLine = [] # list of lists
	equations = [] 
	parsedEquations = []
	variables = []
	ranges = []
	parsedRanges = []
	labels = []

	#reading input and parsing input equations
	readJson(inputPath, parseTreeList, rawLine, equations, variables, ranges, labels, jsonAtts)
	parseEquation(equations, parsedEquations)
	parseEquation(ranges, parsedRanges)
	numSamples = len(parsedEquations)
	buckets = list(xrange(numSamples))
	labels = mx.nd.ones([numSamples,])
	# print equations	
	print "parsedEquations:", parsedEquations[23]
	# print "labels:", labels
	
	samples = []
	dataNames = []
	ctr = 0
	for equation in parsedEquations:
		# treeCounter = count()
		currNNTree = buildTree(treeType=nnTree , parsedEquation=equation, 
			                   num_hidden=num_hidden, params=params, 
			                   emb_dimension=emb_dimension)
		# currNNTree.traverse()

		# state = currNNTree.unroll()

		# print "traversing equation ", ctr
		# currTreeLSTM.traverse()
		# print "travesal done"
		# print state
		currDataNames = currNNTree.getDataNames(dataNames=[])
		# print "currDataNames:", currDataNames
		dataNames.append(currDataNames)
		samples.append(currNNTree)
		# ctr += 1
	# Samples are stored in samples. The data iterator is then only a list iterator. (I think)

	train_eq, _ = encode_equations(parsedEquations, vocab=functionDictionary, invalid_label=-1, invalid_key='\n', start_label=0)
	# data_train = mx.rnn.BucketSentenceIter(train_eq, batch_size)
	data_train  = BucketEqIterator(enEquations=train_eq, eqTreeList=samples, batch_size=batch_size, buckets=buckets, labels=labels,
                                            invalid_label=-1)

	

	# print "parse Length:", len(parsedEquations)
	# print "parsed equations:", parsedEquations
	# print "encoded equations:", train_eq
	# print dataNames

	# print "data_train:", data_train.provide_data
	# print "self index:", data_train.idx
	# print "self current index:", data_train.curr_idx
	# d = data_train.next()
	# print "self current index:", data_train.curr_idx
	# print "data_train:", d.provide_data
	# d = data_train.next()
	# print "self current index:", data_train.curr_idx
	# print "data_train:", d.provide_data
	# d = data_train.next()
	# print "self current index:", data_train.curr_idx
	# print "data_train:", d.provide_data
	# print "data_label:", d.provide_label

	# assert 1==2, "stop"

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
        eval_data           = data_train,
        eval_metric         = mx.metric.Perplexity(0),
        kvstore             = 'str',
        optimizer           = 'sgd',
        optimizer_params    = { 'learning_rate': 0.01,
                                'momentum': 0.0,
                                'wd': 0.00001 },
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch           = 1,
        batch_end_callback  = mx.callback.Speedometer(1, 20))



	# train_eq, _ = encode_equations(parsedEquations, vocab=functionDictionary, invalid_label=-1, invalid_key='\n', start_label=0)


if __name__=='__main__':
	main()








