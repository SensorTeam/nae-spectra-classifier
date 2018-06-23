"""
==================================================================
AUTHOR: HIEN VU
LAST MODIFIED: 20-04-18
==================================================================
Locates the eyes and extracts spectrum from both left and right eyes
Adds eye data to training database with label
INPUT: image (.jpg .png .tiff) containing eyeshine signal, class c
OUTPUT: spectral data in spec.csv
USAGE: execute from terminal
			`python3 main.py -i path-to-image -c class`
==================================================================
"""


from find_eye import *
from find_pairs import *
from get_spectrum import *
import argparse
import matplotlib.pyplot as plt
import csv
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the normal image file")
ap.add_argument("-c", "--class", help = "class for training data")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()
new = image.copy()

# for training data
ID = args["class"]

# find pairs of eyes
contours = find_eye(image)
[con_pairs, pair_det] = find_pairs(image, contours)
num_pairs = len(con_pairs)
fname = os.path.basename(args["image"])
print("------- RESULTS -------")
print("SEARCHED " + str(fname))
print("FOUND " + str(num_pairs) + " PAIR/S")

"""
# set up new databases

# spectrum
fields2 = ['file','ID','L/R']
fields2 = fields2 + list(range(400, 721))
f2 = open("spec.csv", 'w')
writer = csv.writer(f2)
writer.writerow(fields2)
f2.close()
"""

# For each pair
for i in range(0, num_pairs):
	# get eye details for the pair
	con1, con2 = con_pairs[i][0], con_pairs[i][1]
	pair = pair_det[i]
	eye1, eye2 = pair[0], pair[1]

	# get spectrum
	[spec1, spec2] = get_spectrum(pair, orig)

	# graph spectrum
	x1, y1 = zip(*spec1)
	x2, y2 = zip(*spec2)
	plt.plot(x1,y1)
	plt.plot(x2,y2, 'red')
	plt.title("NAE Spectrum - file: %s" % args["image"])
	plt.xlabel("Wavelength (nm)")
	plt.ylabel("Intensity")
	plt.savefig('outputgraph.jpg')
	plt.show()

	# add spectrum to spec database
	f2 = open("spec.csv", 'a')
	writer = csv.writer(f2)
	writer.writerow([fname,' ',ID,'L']+list(y1))
	writer.writerow([fname,' ',ID,'R']+list(y2))
	f2.close()

	
	

