import math
import numpy as np
from matplotlib import pyplot as plt


class KDE:
    def __init__(self, data, ratio, sklearn_valid=False, **args):
        data = data.sample(frac=1).reset_index(drop=True)  # shuffle data
        self.data = data.iloc[:, :-1]
        self.num = data.shape[0]
        self.sklearn_valid = sklearn_valid

    def guassian(self, x):
        return (1/math.sqrt(2*math.pi))*math.exp(-0.5*(x**2))

    def band_width(self, y):
        return 1.05*y.std()*(self.num**(-0.2))

    def kernel_density(self, x, y):
        density = 0
        for i in range(self.num):
            density += self.guassian((x-y[i])/self.band_width(y))
        density /= self.band_width(y) * self.num
        return density, self.band_width(y)

    def run(self):
        band_width_list = []
        fig1 = plt.figure()
        for column in range(self.data.columns.shape[0]):
            density_list = []
            data_column = self.data.iloc[:, column]
            data_sample = np.linspace(data_column.min(), data_column.max(), 50)
            for i in range(data_sample.shape[0]):
                density, band_width = self.kernel_density(data_sample[i], data_column)
                density_list.append(math.log(density))  # return log density
            band_width_list.append(band_width)

            ax2 = fig1.add_subplot(1, self.data.columns.shape[0], column + 1)
            ax2.plot(data_sample, density_list, c='r', marker='.')
            ax2.hist(data_column, density=True)
            ax2.set_title(str(self.data.columns[column]))
        fig1.suptitle('Ours')

        if self.sklearn_valid:
            self.sklearn_check(band_width_list)

        plt.show()

    def sklearn_check(self, band_width_list):
        try:
            from sklearn import neighbors
            print('\n-----Compared with sklearn-----')
            fig2 = plt.figure()
            for column in range(self.data.columns.shape[0]):
                data_column = self.data.iloc[:, column]
                data_sample = np.linspace(data_column.min(), data_column.max(), 50).reshape(-1,1)
                model = neighbors.KernelDensity(kernel='gaussian', bandwidth=band_width_list[column])
                model.fit(data_column.values.reshape(-1,1))

                ax2 = fig2.add_subplot(1, self.data.columns.shape[0], column + 1)
                ax2.plot(data_sample, model.score_samples(data_sample), c='r', marker='.')
                ax2.hist(data_column, density=True)
                ax2.set_title(str(self.data.columns[column]))
            fig2.suptitle('Sklearn')
            plt.show()

        except ImportError:
            print('\nNo package: sklearn')