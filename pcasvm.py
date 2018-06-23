"""
==================================================================
AUTHOR: HIEN VU
LAST MODIFIED: 27-05-18
==================================================================
Performs principal component analysis and support vector machine
classification on NAE spectral data
INPUT: 		spectrain.csv (containing training data)
			spectest.csv (containing test data)
OUTPUT: 	PCA plot
			PCA explained variance ratio
			SVM plots showing decision boundaries four 4 kernels 
			Test data prediction accuracy
USAGE: execute from terminal
			`python3 pcasvm.py`
==================================================================
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from svm import *


############## IMPORT AND CLEAN DATA ####################
# training data
with open('spectrain.csv', 'r') as f:
	reader = csv.reader(f)
	raw = list(reader)
data = np.asarray([row[2:] for row in raw])
y = data[1:, 0]
x = data[1:, 2:].astype(np.float)
idx = [i for i in range(len(y))]

# Import test data
with open('spectest.csv', 'r') as f2:
	reader2 = csv.reader(f2)
	raw = list(reader2)
testdata = np.asarray([row[2:] for row in raw])
ytest = testdata[1:, 0]
xtest = testdata[1:, 2:].astype(np.float)

# Standardising/normalising features (mean=0, std=1)
x = StandardScaler().fit_transform(x)
xtest = StandardScaler().fit_transform(xtest)

##########################################################



########### PRINCIPAL COMPONENT ANALYSIS #################

pca = PCA(n_components=2)
pca.fit(x)		# fit to training data
principalComponents = pca.transform(x)			# transform both test and training data
testprincipalComponents = pca.transform(xtest)
principalDf = pd.DataFrame(data = principalComponents
			 , columns = ['principal component 1', 'principal component 2'])

# Compose new dataframe
targetDf = pd.DataFrame(data=y, index=idx, columns=['target'])
finalDf = pd.concat([principalDf, targetDf], axis = 1)

# Save PCA data to csv
f = open("PCA_2component_train.csv", 'w')
writer = csv.writer(f)
writer.writerow(['principal component 1','principal component 2','target'])
for i in range(len(principalComponents)):
	writer.writerow(principalComponents[i].tolist()+[y[i]])
f.close()

# Save PCA test data to csv
testf = open("PCA_2component_test.csv", 'w')
writer2 = csv.writer(testf)
writer2.writerow(['principal component 1','principal component 2','target'])
for i in range(len(testprincipalComponents)):
	writer2.writerow(testprincipalComponents[i].tolist()+[ytest[i]])
testf.close()

# Plot PCA
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('NAE Spectra PCA (2 Components)', fontsize = 20)

targets = ['orange', 'green', 'blue', 'yellow']
colors = ['r', 'g', 'b', 'y']
for target, color in zip(targets,colors):
	indices = finalDf['target'] == target
	ax.scatter(finalDf.loc[indices, 'principal component 1']
			   , finalDf.loc[indices, 'principal component 2']
			   , c=color, edgecolors='k')
ax.legend(targets)
ax.grid()
plt.savefig("2 component PCA.jpg")
plt.show()

print("PCA EXPLAINED VARIANCE RATIO: %s" % pca.explained_variance_ratio_)	# accounts for how much variance?

##########################################################


############ SUPPORT VECTOR MACHINE ######################

models = pca_svm("PCA_2component_train.csv")
# Prediction for test data
accuracy = predict("PCA_2component_test.csv", models)
print("SVM MODELS PREDICTION ACCURACY: %s" % accuracy)

##########################################################
