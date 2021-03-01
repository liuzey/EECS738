import numpy as np
import random
import math
from matplotlib import pyplot as plt


class GMM:
    def __init__(self, data, ratio, k=3, norm=True, sklearn_valid=False, optimized=True, **args):
        data = data.sample(frac=1).reset_index(drop=True)  # shuffle data
        self.data = data.iloc[:, :-1]
        self.K = k
        self.norm = norm
        self.sklearn_valid = sklearn_valid
        self.optimized = optimized  # Searching for best k among multiple models.
        self.num = self.data.shape[0]
        self.colnum = self.data.columns.shape[0]
        self.max_iter = 100
        self.round = 3
        self.tolerance = 1e-6

    def guassian(self, x_like, mean, cov):
        cov = cov + np.eye(self.colnum) * 1e-5
        diff = (x_like - mean).reshape((1, self.colnum))
        prob = 1.0 / (np.power(np.power(2 * np.pi, self.colnum) * np.abs(np.linalg.det(cov)), 0.5)) * \
               np.exp(-0.5 * diff.dot(np.linalg.inv(cov)).dot(diff.T))[0][0]
        return prob

    def normalize(self, x):
        # z-score normalization.
        x_mean = x.mean()
        x_std = x.std()
        x1 = (x - x_mean) / x_std
        return x1

    def log_likelihood(self, p_list, k):
        log_list = []
        for i in range(self.num):
            log_i = np.array([self.divi[c]*self.guassian(self.data.iloc[i,:].values, self.mean[c], self.cov[c])
                                                        for c in range(k)])
            log_i = np.log(np.array(np.sum(log_i)))
            log_list.append(log_i)
        return np.sum(log_list)

    def E_step(self, p_list, k):
        for i in range(self.num):
            posterior = np.array([self.divi[c]*self.guassian(self.data.iloc[i,:].values, self.mean[c], self.cov[c])
                                  for c in range(k)])
            p_list[i] = posterior/np.sum(posterior)
        return p_list

    def M_step(self, p_list, k):
        for c in range(k):
            num_k = np.sum([p_list[i][c] for i in range(self.num)])
            self.divi[c] = num_k/self.num
            self.mean[c] = np.sum([p_list[i][c]*self.data.iloc[i,:].values for i in range(self.num)], axis=0) / num_k
            diff = self.data.values - self.mean[c]
            self.cov[c] = np.sum([p_list[i][c]*(diff[i].reshape((self.colnum,1)).dot(diff[i].reshape((1,self.colnum)))) for i in range(self.num)], axis=0) / num_k

    def gmm(self, k):
        loglike = 0
        new_loglike = 0
        p_list = [np.zeros(k) for i in range(self.num)]

        self.divi = np.random.rand(k)
        self.divi /= np.sum(self.divi)
        self.mean = [np.random.rand(self.colnum) for c in range(k)]
        self.cov = [np.random.rand(self.colnum, self.colnum) for c in range(k)]

        for i in range(self.max_iter + 1):
            loglike = new_loglike
            p_list = self.E_step(p_list, k)
            self.M_step(p_list, k)
            new_loglike = self.log_likelihood(p_list, k)

            # Exception handle.
            if np.isnan(new_loglike):
                break
            print('Iteration {}, Log-likelihood: {}'.format(i, new_loglike/self.num))

            # Early stop.
            if np.abs(loglike - new_loglike) < self.tolerance:
                print('Ended. Iteration {}, Log-likelihood: {}'.format(i, new_loglike/self.num))
                break

        for p in p_list:
            p /= np.sum(p)

        return p_list, new_loglike

    def multi_seed(self, k):
        # Train model with different initial parameters.
        best_loglike = 0
        best_p_list = {}
        i = 0
        while i < self.round:
            print('k={}, Round: {}'.format(k, i))
            p_list, new_loglike = self.gmm(k)
            if np.isnan(new_loglike):
                continue
            if (i == 0 or new_loglike >= best_loglike):
                best_loglike = new_loglike
                best_p_list = p_list[:]
            i += 1
        print('Best log-likihood: {}'.format(best_loglike/self.num))
        return best_p_list, best_loglike

    def run(self):
        self.data = self.normalize(self.data)
        if not self.optimized:
            best_p_list, best_loglike = self.multi_seed(self.K)
            best_k = self.K
        else:
            best_p_list_list = []
            best_loglike_list = []
            best_bic_list = []
            for i in range(1, 12):
                p_list, new_loglike = self.multi_seed(i)
                best_p_list_list.append(p_list)
                best_loglike_list.append(new_loglike)
                bic = -2 * new_loglike + i * math.log(self.num)
                best_bic_list.append(bic)
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(list(range(1, 12)), best_bic_list, linestyle='-', color='r', marker='o')
            ax1.set_xlabel('k')
            ax1.set_ylabel('bic')
            plt.title('BIC')
            plt.show()

            # Identify best k by 'elbow' method.
            best_k = int(input("Best k you choose: "))
            best_p_list = best_p_list_list[best_k - 1]
            best_loglike = best_loglike_list[best_k - 1]

        print('Our GMM finished. Log-likelihood: {}'.format(best_loglike/self.num))
        clusters = {}
        for c in range(best_k):
            clusters[c] = []
        for i in range(self.num):
            clusters[np.argmax(best_p_list[i])].append(i)

        fig2 = plt.figure()
        # Draw different class in different colors & shapes.
        color = ['r', 'g', 'b', 'k', 'y']
        marker = ['o', '^', '8', 's', 'p', '8', '+']
        for column in range(self.data.columns.shape[0]):
            ax2 = fig2.add_subplot(1, self.data.columns.shape[0], column + 1)
            for k in range(best_k):
                ax2.scatter(self.data.iloc[clusters[k], column],
                            self.data.iloc[clusters[k], column],
                            c=color[k % 5], marker=marker[k // 5], s=20 + 10 * (k % 5))
            ax2.set_title(str(self.data.columns[column]))
        plt.show()

        if self.sklearn_valid:
            self.sklearn_check(best_k)

    def sklearn_check(self, best_k):
        try:
            from sklearn import mixture

            model = mixture.GaussianMixture(n_components=best_k, n_init=3, covariance_type='diag')
            model.fit(self.data)
            print('\n-----Compared with sklearn-----')
            predictions = model.predict(self.data)
            # print(predictions)

            clusters = {}
            for k in range(best_k):
                clusters[k] = []
            for i in range(self.num):
                clusters[predictions[i]].append(i)

            fig = plt.figure()
            # Draw different cluster in different colors & shapes.
            color = ['r', 'g', 'b', 'k', 'y']
            marker = ['o', '^', '8', 's', 'p', '8', '+']
            for column in range(self.data.columns.shape[0]):
                ax2 = fig.add_subplot(1, self.data.columns.shape[0], column + 1)
                for k in range(best_k):
                    ax2.scatter(self.data.iloc[clusters[k], column],
                                self.data.iloc[clusters[k], column],
                                c=color[k % 5], marker=marker[k // 5], s=20 + 10 * (k % 5))
                ax2.set_title(str(self.data.columns[column]))
            plt.title('Sklearn results')
            plt.show()

            print('Log-likelihood using scikit-learn (lower bounds): {}'.format(model.lower_bound_))
        except ImportError:
            print('\nNo package: sklearn')