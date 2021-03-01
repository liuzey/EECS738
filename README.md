# EECS738 Project1 Probably Interesting Data
EECS738 Machine Learning course project1 by Zeyan Liu. Four methods are implemented: k-nearest neighbors (KNN), k-means clustering (KMEANS), kernel density estimation (KDE) and gaussian mixture model (GMM).

## Dataset
All data comes from UCI Machine Learning on Kaggle. Specifically, two classification datasets are chosen:
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
* Scikit-learn is **only** for examine correctness of methods implemented but does not participate. Whether to compare is set [here](https://github.com/liuzey/EECS738/blob/b88d30af51394cd80eecf678254439cb6e0f823c/main.py#L76).

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

### Results Display
Results will be shown successively as follows:
* 
* 
* 

## Algorithms
### Histograms

### K-Nearest Neighbors

### K-means Clustering

### Kernel Density Estimation

### Gaussian Mixture Model


## Schedule
- [x] Theoretical attack for face alignments.
- [ ] Theoretical models for lens tolerance and adjustment.
- [ ] Real parameter acquisition and evasion attack implementation.
- [ ] Theoretical attacks for face recognition (misidentification).
- [ ] Real face recognition attacks.
