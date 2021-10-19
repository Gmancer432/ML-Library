# Contains utility functions common across different classes


# Creates a list of lines from a CSV file
# Each item in the line is kept as a string
# Items may need to be pre-processed before they are sent to a classifier
def ReadCSV(filename):
	lines = []
	with open(filename, 'r') as f:
		for line in f:
			lines.append(line.strip().split(','))
	return lines

# Converts a column to the given datatype
def ConvertColumn(data, columnidx, datatype):
	for d in data:
		d[columnidx] = datatype(d[columnidx])

# Adds an additional column to the data
# Returns the idx of the data column
def AddColumn(data, defaultVal):
    idx = len(data[0])
    for d in data:
        d.append(defaultVal)
    return idx


# Prints a list of lists in csv format
def PrintCSV(data):
	for row in data:
		length = len(row)
		for i in range(length):
			print(row[i], end='')
			if i != length-1:
				print('', end=',')
			else:
				print()
