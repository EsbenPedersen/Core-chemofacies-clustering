# Core-chemofacies-clustering

A Python-based tool for clustering XRF (X-Ray Fluorescence) data using PCA and K-means clustering techniques.

## Overview

This repository provides tools for analyzing and clustering XRF elemental data, focusing on 30 elements from Na through Mo. The analysis pipeline includes:

1. Data Preprocessing
   - Outlier detection
   - Detection limits (LOD) evaluation

2. Analysis
   - Principal Component Analysis (PCA)
   - K-means clustering

## Features

- Manual selection of cluster numbers
- Configurable number of principal components
- Outlier detection and handling
- Comprehensive data output

## Output

The tool generates an enhanced version of your input CSV file with two additional columns:

- `Outliers`: Boolean indicator for analytical outliers
- `Chemofacies`: Cluster assignment for each analysis
  - Regular entries show the assigned chemofacies cluster
  - `NaN` entries indicate outliers (excluded from PCA)

## Data Requirements

Input data should be in CSV format containing XRF measurements for elements Na through Mo.

## Files

- `PCA_chemofacies.ipynb`: Main analysis notebook
- `RandomCore.csv`: Example dataset
- `T5iLOD.csv`: Detection limits reference file