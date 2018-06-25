"""
==================================================================
AUTHOR: HIEN VU
LAST MODIFIED: 27-05-18
==================================================================
Performs support vector machine classification on 2 dimensional 
NAE spectral data
INPUT: 		PCA_2component_train.csv (training data after PCA)
			PCA_2component_test.csv (test data after PCA)
OUTPUT: 	SVM models for 4 kernels (Linear, LinearSVM, Radial 
			Basis Function, Polynomial)
			Prediction accuracy for test data

Modified from scikit-learn.org. Original code available at
http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
==================================================================
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from matplotlib.colors import ListedColormap

GAMMA = 0.01    # RBF
DEGREE = 5      # Polynomial
classes = ['yellow', 'orange', 'green', 'blue']

def make_meshgrid(x, y, h=.05):
	"""Create a mesh of points to plot in

	Parameters
	----------
	x: data to base x-axis meshgrid on
	y: data to base y-axis meshgrid on
	h: stepsize for meshgrid, optional

	Returns
	-------
	xx, yy : ndarray
	"""
	x_min, x_max = x.min() - 1, x.max() + 1
	y_min, y_max = y.min() - 1, y.max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						 np.arange(y_min, y_max, h))
	return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
	"""Plot the decision boundaries for a classifier.

	Parameters
	----------
	ax: matplotlib axes object
	clf: a classifier
	xx: meshgrid ndarray
	yy: meshgrid ndarray
	params: dictionary of params to pass to contourf, optional
	"""
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	out = ax.contourf(xx, yy, Z, **params)
	return out


def pca_svm(filename):
	# open database
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		raw = list(reader)[:]

	data = np.asarray([row for row in raw[1:]])
	y_raw = data[:, 2]
	X = data[:, :2].astype(np.float)
	
	y = []
	for c in y_raw:
		y.append(classes.index(c))
	y = np.asarray(y).astype(np.int)

	# we create an instance of SVM and fit out data. We do not scale our
	# data since we want to plot the support vectors
	C = 1.0  # SVM regularization parameter
	models = (svm.SVC(kernel='linear', C=C),
			  svm.LinearSVC(C=C),
			  svm.SVC(kernel='rbf', gamma=GAMMA, C=C),
			  svm.SVC(kernel='poly', degree=DEGREE, C=C))
	models_fit = (clf.fit(X, y) for clf in models)

	# title for the plots
	titles = ('Model 1: Linear kernel',
			  'Model 2: LinearSVC (linear kernel)',
			  'Model 3: RBF kernel - gamma %s' % GAMMA,
			  'Model 4: Polynomial (degree %s) kernel' % DEGREE)

	# Set-up 2x2 grid for plotting.
	fig, sub = plt.subplots(2, 2)
	plt.subplots_adjust(wspace=0.4, hspace=0.4)

	X0, X1 = X[:, 0], X[:, 1]
	xx, yy = make_meshgrid(X0, X1)

	cmap_light = ListedColormap(['#FFF3AA', '#FFAAAA', '#AAFFAA', '#AAAAFF'])
	cmap_bold = ListedColormap(['#FFDB00', '#FF0000', '#00FF00', '#0000FF'])
	
	for clf, title, ax in zip(models_fit, titles, sub.flatten()):
		plot_contours(ax, clf, xx, yy, cmap=cmap_light, alpha=0.8)
		ax.scatter(X0, X1, c=y, cmap=cmap_bold, s=20, edgecolors='k')
		ax.set_xlim(xx.min(), xx.max())
		ax.set_ylim(yy.min(), yy.max())
		ax.set_xlabel('PC1')
		ax.set_ylabel('PC2')
		ax.set_xticks(())
		ax.set_yticks(())
		ax.set_title(title, fontsize=10)

	plt.suptitle("NAE Spectra SVM Classification")
	plt.savefig("SVM.jpg")
	plt.show()

	return models

# Predict for test data
def predict(filename, models):
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		raw = list(reader)[:]
	data = np.asarray([row for row in raw[1:]])
	actual = data[:, 2]
	X = data[:, :2].astype(np.float)

	# write results to new csv
	g = open("predictions.csv", 'w')
	writer = csv.writer(g)
	writer.writerow(['PC1','PC2', 'model', 'prediction', 'actual'])

	# calculate accuracy for each class
	accuracy = [0]*4
	for i in range(len(X)):
		entry = X[i]
		a = actual[i]
		for j in range(len(models)):
			model = models[j]
			p = model.predict([[entry[0], entry[1]]])
			p = classes[int(p)]
			writer.writerow([entry[0], entry[1], j+1, p, a])
			if p == a:
				accuracy[j] += 1
	g.close()
	accuracy = [i/(len(actual)) for i in accuracy]
	return accuracy
	