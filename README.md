# ML-Library
This is a machine learning library developed by Sean Richens for CS5350/6350 in University of Utah.

-- Decision Tree documentation --

First, read in the data.
Util.ReadCSV(filepath) -> list of rows of data
If the datatype of a column needs to be convered, use 
Util.ConvertColumn(data, columnidx, datatype)

Before training a decision tree, you need to prepare a dictionary of attributes:
{ 'name' : ID3.Attribute(...), ... }
An attribute object describes information the tree needs to know about the attribute:
ID3.Attribute(name, idx, values, numval=False):
- name:    the string name of the attribute
- idx:     the index of the attribute in a row of data
- values:  a list of values the attribute can take, if categorical.  If the attribute is numerical, this list should be empty.
- numval:  defaults to False.  Set to True if the attribute is numerical.

Currently, if you have data with missing values, you need to process that data yourself.

From here, a decision tree can be trained:
	ID3.ID3(data, atrsDict, atrsList, purityfn, maxdepth)
		- data:  a list of rows of data, probably taken from Util.ReadCSV(..)
		- atrsDict:  A prepared dictionary of attributes
		- atrsList:  A list of string names of the attributes
		- purityfn:  The chosen purity function.  Choose from ID3.Entropy, ID3.ME, or ID3.GI
		- maxdepth:  The maximum depth of the tree.  0=root only, None=no limit

Finally, you can calculate error with CalculateError(..)
	CalculateError(data, labelidx, DT)
		- data:  List of rows of data
		- labelidx:  index of the label for the row.  Usually the last item in the row.
		- DT:  The decision tree to calculate error for.