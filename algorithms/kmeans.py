import numpy as np
import random
from matplotlib import pyplot as plt


class KMEANS:
    def __init__(self, data, ratio, k=3, norm=True, sklearn_valid=False, optimized=True, **args):
        data = data.sample(frac=1).reset_index(drop=True)  # shuffle data
        self.data = data.iloc[:, :-1]
        self.K = k
        self.norm = norm
        self.sklearn_valid = sklearn_valid
        self.optimized = optimized  # Searching for best k among multiple models.
        self.num = self.data.shape[0]
        self.round = 10
        self.max_iter = 200
        self.tolerance = 1e-2

    def euclidean_dist(self, input, center):
        # E_D = sum((x-x_i)^2)^(1/2)
        diff = (((input-center) ** 2).sum(axis=0)) ** 0.5
        return diff

    def normalize(self, x):
        # z-score normalization.
        x_mean = x.mean()
        x_std = x.std()
        x1 = (x - x_mean) / x_std
        return x1

    def kmeans(self, center_list, k):
        new_center_list = center_list[:]
        stop = False
        for i in range(self.max_iter + 1):
            errors = 0
            clusters = {}
            tol_list = []

            # Initial clusters.
            for i1 in range(k):
                clusters[i1] = []

            # Cluster each data to the nearest center.
            for j in range(self.num):
                dist_list = []
                for center in new_center_list:
                    dist = self.euclidean_dist(self.data.iloc[j, :], center)
                    dist_list.append(dist)
                errors += min(dist_list)
                clusters[dist_list.index(min(dist_list))].append(j)
            if errors == 0:
                print(dist_list)
                for center in new_center_list:
                    print(center)
                exit()

            # Update centroids.
            for i2 in range(k):
                if len(clusters[i2]) != 0:
                    new_center = self.data.iloc[clusters[i2], :].mean(axis=0)
                    tol_list.append(((new_center-new_center_list[i2])/new_center_list[i2]).abs().sum(axis=0))
                    new_center_list[i2] = new_center

            if i == 0:
                print('Initial Errors: {}'.format(errors))
            else:
                print('Iteration {}, Errors: {}'.format(i-1, errors))

            # Early stop.
            if max(tol_list) <= self.tolerance:
                stop = True
            if stop or i == self.max_iter:
                errors = 0
                for i3 in range(self.num):
                    dist_list = []
                    for center in new_center_list:
                        dist = self.euclidean_dist(self.data.iloc[i3, :], center)
                        dist_list.append(dist)
                    errors += min(dist_list)
                print('Ended. Iteration {}, Errors: {}'.format(i, errors))
                break
        return clusters, errors

    def multi_seed(self, k):
        # Train model with different initial parameters.
        best_errors = 0
        best_clusters = {}
        for i in range(self.round):
            center_list = []
            seed_index = random.sample(list(range(self.num)), k)
            for item in seed_index:
                center_list.append(self.data.iloc[item, :])
            print('k={}, Round: {}'.format(k, i))
            clusters, errors = self.kmeans(center_list, k)
            if i == 0 or errors <= best_errors:
                best_errors = errors
                best_clusters = clusters.copy()
        print('Best errors: {}'.format(best_errors))
        return best_clusters, best_errors

    def run(self):
        self.data = self.normalize(self.data)
        if not self.optimized:
            best_clusters_acrossk, best_errors_acrossk = self.multi_seed(self.K)
            best_k = self.K
        else:
            best_error_list = []
            best_clusters_list = []
            for i in range(1,10):
                best_clusters, best_errors = self.multi_seed(i)
                best_error_list.append(best_errors)
                best_clusters_list.append(best_clusters)
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(list(range(1,10)), best_error_list, linestyle = '-', color='r', marker='o')
            ax1.set_xlabel('k')
            ax1.set_ylabel('errors')
            plt.show()

            # Identify best k by 'elbow' method.
            best_k = int(input("Best k you choose: "))
            best_clusters_acrossk = best_clusters_list[best_k-1]
            best_errors_acrossk = best_error_list[best_k-1]

        print('Our K-means clustering finished. Errors: {}'.format(best_errors_acrossk))
        fig2 = plt.figure()
        # Draw different class in different colors & shapes.
        color = ['r', 'g', 'b', 'k', 'y']
        marker = ['o', '^', '8', 's', 'p', '8', '+']
        for column in range(self.data.columns.shape[0]):
            ax2 = fig2.add_subplot(1, self.data.columns.shape[0], column + 1)
            for k in range(best_k):
                ax2.scatter(self.data.iloc[best_clusters_acrossk[k], column],
                           self.data.iloc[best_clusters_acrossk[k], column],
                           c=color[k % 5], marker=marker[k // 5], s=20+10*(k % 5))
            ax2.set_title(str(self.data.columns[column]))
        plt.show()

        if self.sklearn_valid:
            self.sklearn_check(best_k)

    def sklearn_check(self, best_k):
        try:
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=best_k, random_state=0, n_init=10)
            model.fit(self.data)
            print('\n-----Compared with sklearn-----')
            label = model.labels_
            # print(label)

            fig = plt.figure()
            # Draw different cluster in different colors & shapes.
            color = ['r', 'g', 'b', 'k', 'y']
            marker = ['o', '^', '8', 's', 'p', '8', '+']
            for column in range(self.data.columns.shape[0]):
                ax = fig.add_subplot(1, self.data.columns.shape[0], column+1)
                for k in range(best_k):
                    ax.scatter(self.data.iloc[np.where(label==k)[0].tolist(), column],
                               self.data.iloc[np.where(label==k)[0].tolist(), column],
                                c=color[k % 5], marker=marker[k // 5])
                ax.set_title(str(self.data.columns[column]))
            plt.show()

            print('Errors using scikit-learn: {}%'.format(model.inertia_))
        except ImportError:
            print('\nNo package: sklearn')