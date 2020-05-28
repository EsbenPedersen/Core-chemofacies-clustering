# Core-chemofacies-clustering
PCA and K means to cluster XRF data

XRF elemental data (30 elements Na through Mo) are evalutated for outliers and detection limits (LOD). Data are clustered using Principal Component Analysis (PCA) and K-means clustering. The number of clusters and the number of principal components used to define the clusters are selected manually.

An output .csv file is created that adds two additional columns to the original .csv file. 'Outliers' indicates whether or not that analysis is an analytical outlier. 'Chemofacies' lists the chemofacies cluster that analysis is in. Chemofacies with 'NaN' are outliers and were not included in PCA.   
