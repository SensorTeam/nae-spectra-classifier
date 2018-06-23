"""
==================================================================
AUTHOR: HIEN VU
LAST MODIFIED: 17-06-18
==================================================================
Plots spectral data
INPUT: 		.csv file containing spectral data
OUTPUT: 	graphed spectra
USAGE: execute from terminal
			`python3 plot.py -f path-to-csv-file`
==================================================================
"""

import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Parse argument filename
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help = "path to the .csv file containing spectral data")
args = vars(ap.parse_args())
filename = args["file"]

# Open spectral data
with open(filename, 'r') as f:
	reader = csv.reader(f)
	raw = list(reader)

data = np.asarray([row[2:] for row in raw])
c = data[1:, 0]
x = data[0, 2:].astype(np.int)
y = data[1:, 2:].astype(np.float)

fig = plt.figure(figsize = (8,8))

# Plot each sample
for i in range(len(y)):
	sample = y[i]

	"""
	# plot in the right colour for multiclass dataset
	col = c[i]
	if col == 'yellow':
		pc = 'y'
	elif col == 'green':
		pc = 'g'
	elif col == 'blue':
		pc = 'b'
	elif col == 'orange':
		pc = 'r'
	plt.plot(x,sample,pc)
	"""

	plt.plot(x,sample)

plt.title("Spectrum of NAE Model (training data)", fontsize=20)
plt.xlabel("Wavelength (nm)", fontsize=15)
plt.ylabel("Relative intensity (%)", fontsize=15)
"""
# Legend for multiclass data
leg = [Line2D([0],[0], color='r', label='orange'),
		Line2D([0],[0], color='b', label='blue'),
		Line2D([0],[0], color='g', label = 'green'),
		Line2D([0],[0], color='y', label = 'yellow')]
plt.legend(handles=leg)
"""
#plt.savefig('outputall.jpg')
plt.show()
