# NAE SPECTRA CLASSIFIER
Extracts NAE spectral data from images taken with diffraction grating and classifies using SVM

# Requirements

* `python3`
* `pip`
* `virtualenv`

# Test data
Available at https://goo.gl/xV2AWD

# Usage
1. Setup a virtual environment using the command `virtualenv venv`
2. Activate the virtual environment using `source venv/bin/activate`
3. Install package dependencies using `pip install -r requirements.txt`
4. Execute using the following

## Obtaining calibration values from known wavelength light source
	`python3 calib.py -f path-to-image`
		Test file: data/calibration.jpg
(NB: To obtain calibration values, using output files and look for peaks 
at known wavelengths. For example, for a mercury lamp, use peak at 
546.1nm. Take corresponding pixel value to input for get_spectrum.py)

## Extracting spectra from an image
	`python main.py -i path-to-image -c class`
		Test file: data/sample.tiff, data/class=blue

## Plotting obtained spectra
	`python3 plot.py -f path-to-csv-file`
		Test files: data/spec.csv, data/specblue.csv

## Splitting data into training and test data
	`python3 split.py`

## PCA and SVM on dataset
	`python3 pcasvm.py`
