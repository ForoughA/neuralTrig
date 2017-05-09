import mxnet as mx
import numpy as np 
from neuralAlgonometry import readJsonEquations, makeEqTree, buildNNTree, encode_equations, get_indices, EquationTree
from nnTreeMain import nnTree, BucketEqIterator
import random
import argparse

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
# functionVocab.extend(functionOneInp)
# functionVocab.extend(functionTwoInp)
# functionVocab.extend(functionBothSides) 
# tmp.extend(functionVocab)
tmp.extend(variables) 
tmp.extend(consts)

functionDictionary = {}
ctr = 1
for f in tmp:
	functionDictionary[f] = ctr
	ctr+=1
################################################################################

def main():

	import logging
	head = '%(asctime)-15s %(message)s'
	logging.basicConfig(level=logging.DEBUG, format=head)

	## NN atts:
	params = None
	contexts = mx.cpu(0)
	num_hidden = 120
	vocabSize = len(functionDictionary)
	emb_dimension = 16
	out_dimension = 1
	batch_size = 1
	split = 0.8

	path = 'data.json'
	[equations, eqVariables, ranges, labels] = readJsonEquations(path)

	numSamples = len(equations)
	buckets = list(xrange(numSamples))

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
	# random.shuffle(indices)
	splitInd = int(split * len(equations))
	trInd = indices[:splitInd]
	teInd = indices[splitInd:]
	trainEquations = [equations[i] for i in trInd]
	testEquations = [equations[i] for i in teInd]
	trainLabels = [labels[i] for i in trInd]
	testLabels = [labels[i] for i in teInd]
	trainVars = [eqVariables for i in trInd]
	testVars = [eqVariables for i in teInd]

	# print "sample 0 preOrder"
	# equations[0].preOrder()
	# print "sample 1 preOrder"
	# equations[1].preOrder()


	# train_eq, _ = encode_equations(equations, vocab=functionDictionary, invalid_label=-1, invalid_key='\n', start_label=0)
	all_eq, _ = encode_equations(equations, vocab=functionDictionary, invalid_label=-1, invalid_key='\n', start_label=0)
	train_eq, _ = encode_equations(trainEquations, vocab=functionDictionary, invalid_label=-1, invalid_key='\n', start_label=0)
	test_eq, _ = encode_equations(testEquations, vocab=functionDictionary, invalid_label=-1, invalid_key='\n', start_label=0)
	# data_train = mx.rnn.BucketSentenceIter(train_eq, batch_size)
	# data_train  = BucketEqIterator(enEquations=train_eq, eqTreeList=samples, batch_size=batch_size, buckets=buckets, labels=labels,
 #                                            invalid_label=-1)
	data_all  = BucketEqIterator(enEquations=all_eq, eqTreeList=samples, batch_size=batch_size, buckets=buckets, labels=labels,
                                            invalid_label=-1)
	data_train  = BucketEqIterator(enEquations=train_eq, eqTreeList=samples, batch_size=batch_size, buckets=buckets, labels=trainLabels,
                                            invalid_label=-1)
	data_test  = BucketEqIterator(enEquations=test_eq, eqTreeList=samples, batch_size=batch_size, buckets=buckets, labels=testLabels,
                                            invalid_label=-1)

	cell = samples[0]

	def sym_gen(bucketIDX):
		# print "in sym_gen"
		data = mx.sym.Variable('data')
		label = mx.sym.Variable('softmax_label')
		# embed = mx.sym.Embedding(data=data, input_dim=len(functionVocab),
		#                          output_dim=args.num_embed, name='embed')

		
		# We need to figure out how to use the bucketIDX. 
		# I think the original one handles it using the data iterator.
		# We might be able to handle this using mx.rnn.BucketSentenceIter

		#consider reseting
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

		# label = mx.sym.Reshape(label, shape=(-1,))
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
        kvstore             = 'local',
        eval_metric         = 'acc',
        optimizer           = 'sgd',#mx.optimizer.Adam(beta1=0.9, beta2=0.999, epsilon=1e-08),
        optimizer_params    = { 'learning_rate': 0.0001,
        						'momentum' : 0.0,
                                'wd': 0.0001 },
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch           = 8)
        # epoch_end_callback  = mx.rnn.do_rnn_checkpoint(cell, 'trainedModel', 1))
        # batch_end_callback  = mx.callback.Speedometer(1, 1))

	# modelTest = mx.mod.BucketingModule(
	# 	sym_gen             = sym_gen,
	# 	default_bucket_key  = 0,
	# 	context             = contexts)
	
	# modelTest.bind(data_test.provide_data, data_test.provide_label, for_training=False)

	# _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(cell, 'trainedModel', 8)
	# modelTest.set_params(arg_params, aux_params)

	# metric = mx.metric.Accuracy()
	# print modelTest.score(data_test, metric)


	# predicts = modelTest.predict(data_test).asnumpy()
	# predicts = np.rint(predicts)
	# print "number of 1's:", np.sum(predicts)
	# print "pred shape:", predicts.shape

	# tstLs = [testLabels[i].asnumpy() for i in range(len(testLabels))]
	# tstLs = np.array(tstLs)
	# print "tstLs shape", tstLs.shape
	# print "accuracy:", float(sum(predicts==tstLs))/len(testLabels)



if __name__ == "__main__":
	main()