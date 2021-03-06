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

# Saves CSV data to a file, with an optional header list
def SaveCSV(data, filepath, headerlist=None):
    with open(filepath, 'w') as f:
        if headerlist != None:
            length = len(headerlist)
            for i in range(length):
                f.write(str(headerlist[i]))
                if i != length-1:
                    f.write(',')
                else:
                    f.write('\n')
        for row in data:
            length = len(row)
            for i in range(length):
                f.write(str(row[i]))
                if i != length-1:
                    f.write(',')
                else:
                    f.write('\n')

# Converts a column to the given datatype
def ConvertColumn(data, columnidx, datatype):
	for d in data:
		d[columnidx] = datatype(d[columnidx])

# Adds an additional column to the data
# Returns the idx of the data column
def AddColumn(data, defaultVal):
    idx = len(data[0])
    for i in range(len(data)):
        data[i].append(defaultVal)
    return idx

# Returns a true copy of a list of lists
def CopyData(data):
    newdata = data.copy()
    for i in range(len(data)):
        newdata[i] = data[i].copy()
    return newdata


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
