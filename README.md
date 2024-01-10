# Bias Detection in Open Building Datasets

## Overview
This project aims to detect biases in open building datasets. It includes a suite of Python scripts designed to analyze building dataset accuracy, visualize data through barplots, and conduct correlation and regression analysis.

## Files Included
1. `barplots.py`
2. `buildings_accuracy.py`
3. `correlation_regression.py`

## Usage

### `buildings_accuracy.py`
This script conducts an accuracy analysis on open building datasets. 

**Inputs:**
- At least one building dataset for analysis.
- A reference building dataset (ground-truth).
- A file specifying study areas, which includes sensitive variables.

**Features:**
- Calculates true positives, false positives, and false negatives using the IoU method (default threshold: 0.5, adjustable).
- Outputs Excel files for each building dataset at both building and study area levels.

**Customization:**
- Specify pathfiles to your datasets.
- Adjust column names for sensitive variables in the study area file in the 'append results' section.

### `barplots.py`
This script generates barplots based on the Excel files created by `buildings_accuracy.py`.

**Input:**
- Excel files on tile level from `buildings_accuracy.py`.

**Output:**
- Barplots comparing each sensitive variable to the false negative rate (or other fairness metrics) across all building datasets.
- Equality of opportunity calculation for each variable.

### `correlation_regression.py`
Performs correlation analysis and weighted linear regression.

**Input:**
- An Excel file on tile level.

**Features:**
- Conducts correlation analysis between sensitive variables and the false negative rate (or other metrics).
- Performs weighted linear regression (default weight: building density, adjustable).
- Prints correlation results and displays significant linear regression results through scatter plots.
