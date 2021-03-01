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
* In optimized KMeans and GMM, a graph of K with criterion will be given. A command input has to be given judged on 'elbow method'.

## Study 1: Zoo Dataset
### Histogram
* Histograms are basic represenations of data frequency in bins of values.

![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Zoo_hist.png)

### KNN
```bash
python main.py zoo KNN -k 3
```
* A data point is labelled by voting of K nearest neighbors in distance.
* Euclidean metrics are used.
* A quick glance at distribution:

![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Zoo_KNN.png)
* Accuracy when K=3: 19/21=90.48%.
* Results using scikit-learn when K=3: 19/21=90.48%.

### Kernel Density Estimation
```bash
python main.py zoo KDE
```
* Fit each data point with a family of kernel functions. Stack functions together to represent the whole dataset.
* Guassian kernels are used.

![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Zoo_KDE_my.png)
* Results using scikit-learn are the same. [here](https://github.com/liuzey/EECS738/blob/main/saved_fig/Zoo_KDE_skl.png)

### K-means Clustering
* Assign each data point to the nearest cluster center. Centers are updated as average of data points in each cluster.
* Errors are calculated as the sum of Euclidean distances.
* When **--optimized=False**, K (number of clusters) is set at total number of labels mannually.

```bash
python main.py zoo KMEANS -k 7
```
* When **--optimized=True**, K is optimized across [1,9]. Elbow method is used to determine the best K.

```bash
python main.py zoo KMEANS -o 1
```
![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Zoo_KMEANS_elbow.png)
![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Zoo_KMEANS_my.png)
* Results using scikit-learn can be find [here](https://github.com/liuzey/EECS738/blob/main/saved_fig/Zoo_KMEANS_skl.png).

### Gaussian Mixture Model
* Several Gaussian distribution are used to represent the dataset. Each datapoint is assigned to the 'nearest distribution' based on posterior probability. Thus, data points following different distributions are clustered.
* When **--optimized=False**, K (number of clusters) is set at total number of labels mannually.

```bash
python main.py zoo GMM -k 7
```
* When **--optimized=True**, K is optimized across [2,10]. Bayesian Information Criterion(BIC) is used to model performance and cost. Elbow method is used to determine the best K.

```bash
python main.py zoo GMM -o 1
```
![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Zoo_GMM_bic.png)
![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Zoo_GMM_my.png)
* Results using scikit-learn can be find [here](https://github.com/liuzey/EECS738/blob/main/saved_fig/Zoo_GMM_skl.png).

## Study 2: Glass Dataset
Settings aligns with Zoo Dataset.
### Histogram
![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Glass_hist.png)

### KNN
* A quick glance at distribution:

![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Glass_KNN.png)
* Accuracy when K=3: 24/43=55.81%, K=8: 29/43=67.44%.
* Results using scikit-learn when K=3: 26/43=60.47%, K=8: 29/43=67.44%.

### Kernel Density Estimation
![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Glass_KDE_my.png)
* Results using scikit-learn are the same. [here](https://github.com/liuzey/EECS738/blob/main/saved_fig/Glass_KDE_skl.png)

### K-means Clustering
![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Glass_KMEANS_elbow.png)
![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Glass_KMEANS_my.png)
* Results using scikit-learn can be find [here](https://github.com/liuzey/EECS738/blob/main/saved_fig/Glass_KMEANS_skl.png).

### Gaussian Mixture Model
![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Glass_GMM_bic.png)
![](https://github.com/liuzey/EECS738/blob/main/saved_fig/Glass_GMM_my.png)
* Results using scikit-learn can be find [here](https://github.com/liuzey/EECS738/blob/main/saved_fig/Glass_GMM_skl.png).

## Notes
* KNN doesn't behave well for Glass Dataset. Reasons may be that the dataset is unbalanced.
* Performance of KNN improves with K to avoid overfitting noises.


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
