import numpy as np
from matplotlib import pyplot as plt


class KNN:
    def __init__(self, data, ratio, k=3, norm=True, sklearn_valid=False, **args):
        data = data.sample(frac=1).reset_index(drop=True)  # shuffle data
        self.data = data.iloc[:, :-1]
        self.labels = data.iloc[:, -1]
        self.K = k
        self.norm = norm
        self.sklearn_valid = sklearn_valid
        self.ratio = 1-ratio
        self.num = self.data.shape[0]

    def euclidean_dist(self, input, dataset):
        # E_D = sum((x-x_i)^2)^(1/2)
        data_num = dataset.shape[0]
        diff = (((np.tile(input, (data_num, 1)) - dataset) ** 2).sum(axis=1)) ** 0.5
        return diff

    def normalize(self, x):
        # z-score normalization.
        x_mean = x.mean()
        x_std = x.std()
        x1 = (x - x_mean) / x_std
        return x1

    def run(self):
        norm_data = self.normalize(self.data)
        # norm_data = self.data

        # training-validation data split
        train_d = norm_data.iloc[:int(self.num * self.ratio), :]
        test_d = norm_data.iloc[int(self.num * self.ratio):, :]
        train_l = self.labels[:int(self.num * self.ratio)]
        test_l = self.labels[int(self.num * self.ratio):]
        test_num = test_d.shape[0]

        count = 0
        for index, item in test_d.iterrows():
            dist = self.euclidean_dist(item, train_d).argsort()
            # print(train_l[dist[:self.K].values].value_counts())
            # print(test_l[index])
            item_label = train_l[dist[:self.K].values].value_counts().index[0]
            if item_label == test_l[index]:
                count += 1
        print('(k={}) Our Accuracy: {}/{}={}%'.format(self.K, count, test_num, 100 * count / test_num))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        mapping = {}
        labels = self.labels.value_counts().index
        for i in range(len(labels)):
            mapping[labels[i]] = i
        # print(mapping)
        # Draw different class in different colors & shapes.
        color = ['r', 'g', 'b', 'k', 'y']
        marker = ['o', '^', '8', 's', 'p', '8', '+']
        for k in range(self.num):
            ax.scatter(self.data.iloc[k,0], self.data.iloc[k,1], self.data.iloc[k,2],
                       c=color[(mapping[self.labels[k]])%5], marker=marker[(mapping[self.labels[k]])//5])
        ax.set_xlabel(self.data.columns[0])
        ax.set_ylabel(self.data.columns[1])
        ax.set_zlabel(self.data.columns[2])
        plt.title('Data Distribution')
        plt.show()

        if self.sklearn_valid:
            self.sklearn_check(train_d, test_d, train_l, test_l)

    def sklearn_check(self, train_d, test_d, train_l, test_l):
        try:
            from sklearn import neighbors

            model = neighbors.KNeighborsClassifier(self.K)
            model.fit(train_d, train_l)
            accuracy = model.score(test_d, test_l)
            print('\n-----Compared with sklearn-----')
            print('Accuracy using scikit-learn: {}%'.format(100 * accuracy))
        except ImportError:
            print('\nNo package: sklearn')


