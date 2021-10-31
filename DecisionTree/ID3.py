
# ID3.py
# Created by Sean Richens for UofU CS5350
# --------
# Contains methods for constructing and using a decision tree.

import math
import statistics
import random

# Data needed to process attributes
# name: the name of the attributes
# idx: the index of this attribute as it appears in data
# values: list of possible values (if finite)
# numval: True if this value should be treated as a number value.
class Attribute:
	def __init__(self, name, idx, values, numval=False):
		self.name = name
		self.idx = idx
		self.values = values
		self.numval = numval


# A category node in the decision tree
# Data is passed to the first child with a matching value
# atridx: the index of attribute to match within the data row
class DecisionTreeCat:
	def __init__(self, atridx):
		self.atridx = atridx
		# categories are tuples of (value, child)
		self.cats = []
	# Adds a child with a corresponding attribute value
	def AddChild(self, val, child):
		self.cats.append((val, child))
	# Returns a label by calling PredictLabel on a child
	def PredictLabel(self, data):
		dataval = data[self.atridx]
		for val, child in self.cats:
			if val == dataval:
				return child.PredictLabel(data)

# A numerical node in the decision tree
# Data is passed to the child on the appropriate side of the threshold
# atridx:   the index of the attribute to compare
# thresh:   the numerical threshold
# gtechild: next child if the value is >= the threshold
# ltchild:  next child if the value is < the threshold
class DecisionTreeNum:
	def __init__(self, atridx, thresh):
		self.atridx = atridx
		self.thresh = thresh
		self.gtechild = None
		self.ltchild = None
	# Next child if the value is >= the threshold
	def AddGteChild(self, gtechild):
		self.gtechild = gtechild
	# Next child if the value is < the threshold
	def AddLtChild(self, ltchild):
		self.ltchild = ltchild
	# Returns a label by calling PredictLabel on a child
	def PredictLabel(self, data):
		dataval = data[self.atridx]
		if dataval >= self.thresh:
			return self.gtechild.PredictLabel(data)
		else:
			return self.ltchild.PredictLabel(data)

# A leaf node in the decision tree
# Returns a label for the data
class DecisionTreeLeaf:
	def __init__(self, label):
		self.label = label
	# Returns the label
	# the data parameter is just a formality that allows abstraction
	def PredictLabel(self, data):
		return self.label

# Gets counts of each value for the attribute
# Attribute must be the category type
def CountAttributeCat(data, attribute, atrsDict, weightidx):
	# Set up the buckets
	labelAtr = atrsDict[attribute]
	valueCounts = {}
	for v in labelAtr.values:
		valueCounts[v] = 0
	# Count up how many there are of each label
	total = 0
	for d in data:
		val = d[labelAtr.idx]
		if weightidx >= len(d) or weightidx < 0:
			print(attribute)
			print(len(d))
			print(d)
			print(weightidx)
		w = 1 if weightidx == None else d[weightidx]
		valueCounts[val] += w
		total += w
	# Return the counts
	return (list(valueCounts.values()), total)

# Gets the median of an attribute for a dataset
# Attribute must be the numerical type
def FindThreshold(data, attribute, atrsDict):
	labelAtr = atrsDict[attribute]
	vallist = []
	for d in data:
		vallist.append(d[labelAtr.idx])
	median = statistics.median(vallist)
	return median

# Gets counts of each value for the attribute
# Attribute must be the numerical type
def CountAttributeNum(data, attribute, atrsDict, thresh, weightidx):
	# Set up the buckets
	labelAtr = atrsDict[attribute]
	valueCounts = [0, 0]
	# Count up how many there are of each label
	total = 0
	for d in data:
		value = d[labelAtr.idx]
		w = d[weightidx]
		if value >= thresh:
			valueCounts[0] += w
		else:
			valueCounts[1] += w
		total += w
	# Return the counts
	return (list(valueCounts.values), total)


# Calcuates the Entropy of the data
# ValueCounts: a count of each category
# total: the total across all categories
def Entropy(valueCounts, total):
	entropy = 0
	if total == 0:
		return 0
	for c in valueCounts:
		if c != 0:
			entropy -= c / total * math.log(c / total)
	return entropy

# Calculates the Majority Error of the data
# ValueCounts: a count of each category
# total: the total across all categories
def ME(valueCounts, total):
	if total == 0:
		return 0
	maxval = max(valueCounts)
	maxidx = valueCounts.index(maxval)
	error = 0
	valueCounts.pop(maxidx)
	for c in valueCounts:
		error += c / total
	return error

# Calculates the Gini Index of the data
# ValueCounts: a count of each category
# total: the total across all categories
def GI(valueCounts, total):
	if total == 0:
		return 0
	gi = 1
	for c in valueCounts:
		gi -= (c / total)**2
	return gi

# Splits data by an attribute
# Data must be a category type
# Returns:
#	subsets: a dictionary of value : list of data
#	counts:  a dictionary of value : data count
#	total:	 count of all data
def SplitDataCat(data, attribute, atrsDict):
	atr = atrsDict[attribute]
	vals = atr.values
	subsets = {}
	counts = {}
	total = 0
	for v in vals:
		subsets[v] = []
		counts[v] = 0
	for d in data:
		atrval = d[atr.idx]
		subsets[atrval].append(d)
		counts[atrval] += 1
		total += 1
	return (subsets, counts, total)

# Splits data by an attribute
# Data must be a numerical type
# Returns:
#	subsets: list of lists of data
#			 [0] = gte, [1] = lt
#	counts:  list of data counts
#	total:	 count of all data
def SplitDataNum(data, attribute, atrsDict, thresh):
	atr = atrsDict[attribute]
	subsets = [[], []]
	counts = [0, 0]
	total = 0
	for d in data:
		atrval = d[atr.idx]
		if atrval >= thresh:
			subsets[0].append(d)
			counts[0] += 1
			total += 1
		else:
			subsets[1].append(d)
			counts[1] += 1
			total += 1
	return (subsets, counts, total)

# Recursively creates a DecisionTree based on the inputs
# data:       list of rows of data
# weightidx:  the index of the weight of an example in a data row
# atrsDict:   dictionary of Attribute objects
# atrsList:   list of attributes to choose from
# purityfn:   function that inputs data and outputs a 'purity' measure
#             (valueCounts, total) => float
#             Use one of the implementations of Entropy, ME, or GI
# maxdepth:   the maximum depth of the tree (0=root only, None=no limit)
# maxatrs:    the maximum amount of attributes to pick from at each node.
#             used in bagging.  maxatrs=None => no limit
def ID3(data, weightidx, atrsDict, atrsList, purityfn, maxdepth, maxatrs=None):
	# Base Cases: If all have the same label, return a leaf node
	#             If atrsList is empty, return a leaf node with the most common label
	#			  If maxdepth==0, return a leaf node with the most common label
	labelCounts, totalCount = CountAttributeCat(data, 'label', atrsDict, weightidx)
	maxCount = max(labelCounts)
	labellist = atrsDict['label'].values
	majoritylabel = labellist[labelCounts.index(maxCount)]
	if maxCount == totalCount or len(atrsList) == 0 or (maxdepth != None and maxdepth <= 0):
		return DecisionTreeLeaf(majoritylabel)
	# Update the depth
	if maxdepth != None:
		maxdepth -= 1
	# Find the attribute with the best information gain
	purity = purityfn(labelCounts, totalCount)
	curbestgain = 0
	curbestattribute = None # attribute object
	curdatasubsets = None
	curbestcounts = None
	bestthresh = 0
	curatrsList = atrsList
	if maxatrs != None:
        maxatrs = min(maxatrs, len(atrsList))
		curatrsList = random.sample(atrsList, maxatrs)
	for a in curatrsList:
		attribute = atrsDict[a]
		atridx = attribute.idx
		curgain = purity
		# For category attributes
		if not attribute.numval:
			# Split the data
			subsets, counts, total = SplitDataCat(data, a, atrsDict)
			# Calculate information gain
			vals = attribute.values
			for v in vals:
				subsetlabelCounts, subsettotal = CountAttributeCat(subsets[v], 'label', atrsDict, weightidx)
				subsetpurity = purityfn(subsetlabelCounts, subsettotal)
				curgain -= subsettotal / total * subsetpurity
			# Check if this attribute is the best
			if curgain > curbestgain or curbestattribute == None:
				curbestgain = curgain
				curbestattribute = attribute
				curdatasubsets = subsets
				curbestcounts = counts
		# For numerical attributes
		else:
			# Split the data
			thresh = FindThreshold(data, a, atrsDict)
			subsets, counts, total = SplitDataNum(data, a, atrsDict, thresh)
			# Calculate information gain
			# gte
			gtelabelCounts, gtetotal = CountAttributeCat(subsets[0], 'label', atrsDict, weightidx)
			gtepurity = purityfn(gtelabelCounts, gtetotal)
			curgain -= gtetotal / total * gtepurity
			# lt
			ltlabelCounts, lttotal = CountAttributeCat(subsets[1], 'label', atrsDict, weightidx)
			ltpurity = purityfn(ltlabelCounts, lttotal)
			curgain -= lttotal / total * ltpurity
			# Check if this attribute is the best
			if curgain > curbestgain or curbestattribute == None:
				curbestgain = curgain
				curbestattribute = attribute
				curdatasubsets = subsets
				curbestcounts = counts
				bestthresh = thresh
	# Create the current node and add children
	newatrsList = atrsList.copy()
	newatrsList.pop(newatrsList.index(curbestattribute.name))
	if curbestattribute.numval:
		node = DecisionTreeNum(curbestattribute.idx, bestthresh)
		gtechild = None
		# if the subset is empty, create a leaf with the most common label
		if curbestcounts[0] == 0:
			gtechild = DecisionTreeLeaf(majoritylabel)
		else:
			gtechild = ID3(curdatasubsets[0], weightidx, atrsDict, newatrsList, purityfn, maxdepth)
		node.AddGteChild(gtechild)
		ltchild = None
		# if the subset is empty, create a leaf with the most common label
		if curbestcounts[1] == 0:
			ltchild = DecisionTreeLeaf(majoritylabel)
		else:
			ltchild = ID3(curdatasubsets[1], weightidx, atrsDict, newatrsList, purityfn, maxdepth)
		node.AddLtChild(ltchild)
	else:
		node = DecisionTreeCat(curbestattribute.idx)
		atrvals = curbestattribute.values
		for v in atrvals:
			child = None
			# if the subset is empty, create a leaf with the most common label
			#print(curbestattribute.name)
			#print(curbestcounts)
			if curbestcounts[v] == 0:
				child = DecisionTreeLeaf(majoritylabel)  
			else:
				child = ID3(curdatasubsets[v], weightidx, atrsDict, newatrsList, purityfn, maxdepth)
			node.AddChild(v, child)
	return node
	

# Runs through the data, generates labels, and calculates the error of the predicions
def CalculateError(data, labelidx, DT, weightidx):
	total = 0
	incorrect = 0
	for d in data:
		w = 1 if weightidx == None else d[weightidx]
		total += w
		outlabel = DT.PredictLabel(d)
		if outlabel != d[labelidx]:
			incorrect += w
	return incorrect / total
	
		
	










	
		
		
		
		
		




