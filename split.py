"""
==================================================================
AUTHOR: HIEN VU
LAST MODIFIED: 14-06-18
==================================================================
SplitS data into test and training data (4:1) for SVM
INPUT: 		spec.csv (containing all data)
OUTPUT: 	spectest.csv (containing test data)
			spectrain.csv (containing training data)
USAGE: execute from terminal
			`python3 split.py`
==================================================================
"""

# open file
file = open("spec.csv", "r")
data = file.readlines()

# open output files
test = open("spectest.csv", "a")
train = open("spectrain.csv", "a")

# header row
test.write(data[0])
train.write(data[0])

# separate every 4th entry to test data
for i in range(1, len(data)):
	if i%5 == 0:
		test.write(data[i])
	else:
		train.write(data[i])

file.close()
test.close()
train.close()
