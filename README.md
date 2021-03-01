# EECS738 Project1 Probably Interesting Data
EECS738 Machine Learning course project1 by Zeyan Liu. Four methods are implemented: k-nearest neighbors (KNN), k-means clustering (KMEANS), kernel density estimation (KDE) and gaussian mixture model (GMM).

## Dataset
All data comes from UCI Machine Learning on Kaggle. Specifically, **two** classification datasets are chosen:
* [Zoo Animal Classification.](https://www.kaggle.com/uciml/zoo-animal-classification)
* [Glass Classification.](https://www.kaggle.com/uciml/glass)

If you'd like to add your own data, please put them under ['data'](https://github.com/liuzey/EECS738/tree/main/data) to ensure the hierachy goes like './data/dataset_name/data_file.csv'. Also, put the label column on the rightmost.

## Motivation & Ideas
Each data point combines with numerical features. After normalization, we can easily use kernel function to estimate the density, or use distance or Bayesian statistics to cluster them.

## Setup
### Environment
* Python 3.8
* MacOS or Linux

### Package
* Recommend setting up a virtual environment. Different versions of packages may cause unexpected failures.
* Install packages in **requirements.txt**.
```bash
pip install -r requirements.txt
``` 
* **Scikit-learn is only for examine correctness of methods implemented but does not participate.** Whether to compare is set [here](https://github.com/liuzey/EECS738/blob/b88d30af51394cd80eecf678254439cb6e0f823c/main.py#L76).

## Usage
### Positional & Optional Parameters
* **data**: Data dir path, e.g. 'iris', 'zoo' or 'glass'.
* **algorithm**: Four algorithms to choose from: KNN, KDE, KMEANS, GMM.
* **-k**: Number of clusters in K_means and GMM. Or number of neighbors in KNN. (Default: 3).
* **-r**: Ratio of dataset for validation. (0.0~1.0) (Default: 0.2).
* **-o, --optimized**: If set true, cluster number will instead be optimized in K-means and GMM. (Default: False.).

### Example
```bash
python main.py glass KMEANS -k 7 -o False
```
* Apply K_means algorithm to 'glass' dataset.
* Number of clusters manually set at 7.

### Running Display
Results will be shown successively as follows:
* Histograms, scattered matrix between features and box plots.
* My algorithm results (including visualization).
* Results (including some visualization) for comparison from scikit-learn.

## Zoo
### Histogram
* Histograms are basic represenations of data frequency in bins of values.
![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Zoo_hist.png)

### KNN
* A data point is labelled by voting of k nearest neighbors in distance.
* Euclidean metrics are used.
* A quick glance at distribution:
![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Zoo_KNN.png)
* Accuracy when k=3: 19/21=90.48%
* Results from scikit-learn when k=3: 19/21=90.48%

### Kernel Density Estimation
* Fit each data point with a family of kernel functions. Stack functions together to represent whole dataset.
* Guassian kernels are used.
![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Zoo_KDE.png)

### K-means Clustering

### Gaussian Mixture Model


## Schedule
- [x] Set up a new git repository in your GitHub account.
- [x] Pick two datasets from (https://www.kaggle.com/uciml/datasets).
- [x] Choose a programming language (Python, C/C++, Java). **Python**
- [x] Formulate ideas on how machine learning can be used to model distributions within the dataset.
- [x] Build a heuristic and/or algorithm to model the data using mixture models of probability distributions programmatically.
- [x] Document your process and results.
- [x] Commit your source code, documentation and other supporting files to the git repository in GitHub.

## Reference
* Histogram - Wikipedia. https://en.wikipedia.org/wiki/Histogram
* GaussianMixture - scikit-learn. https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
* KMeans - scikit-learn. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
* KernelDensity - scikit-learn. https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
* Scikit-learn - Github. https://github.com/scikit-learn/scikit-learn/blob/a95203b/sklearn/mixture/gmm.py
* https://medium.com/@analyttica/what-is-bayesian-information-criterion-bic-b3396a894be6
* https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
* Matplotlib APIs. https://matplotlib.org/3.1.1/api/index.html
* Pandas APIs. https://pandas.pydata.org/pandas-docs/stable/reference/frame.html
