import numpy
import random


class Softmax:
    """Softmax regression"""
    def __init__(self, sample, typenum, feature):
        self.sample = sample  # How many sample in training set
        self.typenum = typenum  # How many categories
        self.feature = feature  # The size of feature vector
        self.W = numpy.random.randn(feature, typenum)  # Weight matrix initialization

    def softmax_calculation(self, x):
        """Calculate softmax function value. x is a vector."""
        exp = numpy.exp(x - numpy.max(x))
        return exp / exp.sum()

    def softmax_all(self, wtx):
        """Calculate softmax function value. wtx is a matrix."""
        wtx -= numpy.max(wtx, axis=1, keepdims=True)
        wtx = numpy.exp(wtx)
        wtx /= numpy.sum(wtx, axis=1, keepdims=True)
        return wtx

    def change_y(self, y):
        """Transform an 'int' into a one-hot vector."""
        ans = numpy.array([0] * self.typenum)
        ans[y] = 1
        return ans.reshape(-1, 1)

    def prediction(self, X):
        """Given X, predict the category."""
        prob = self.softmax_all(X.dot(self.W))
        return prob.argmax(axis=1)

    def correct_rate(self, train, train_y, test, test_y):
        """Calculate the categorization accuracy."""
        # train set
        n_train = len(train)
        pred_train = self.prediction(train)
        train_correct = sum([train_y[i] == pred_train[i] for i in range(n_train)]) / n_train
        # test set
        n_test = len(test)
        pred_test = self.prediction(test)
        test_correct = sum([test_y[i] == pred_test[i] for i in range(n_test)]) / n_test
        print(train_correct, test_correct)
        return train_correct, test_correct

    def regression(self, X, y, alpha, times, strategy="mini", mini_size=100):
        """Softmax regression"""
        if self.sample != len(X) or self.sample != len(y):
            raise Exception("Sample size does not match!")
        if strategy == "mini":
            for i in range(times):
                increment = numpy.zeros((self.feature, self.typenum))  # The gradient
                for j in range(mini_size):  # Choose a mini-batch of samples
                    k = random.randint(0, self.sample - 1)
                    yhat = self.softmax_calculation(self.W.T.dot(X[k].reshape(-1, 1)))
                    increment += X[k].reshape(-1, 1).dot((self.change_y(y[k]) - yhat).T)
                # print(i * mini_size)
                self.W += alpha / mini_size * increment
        elif strategy == "shuffle":
            for i in range(times):
                k = random.randint(0, self.sample - 1)  # Choose a sample
                yhat = self.softmax_calculation(self.W.T.dot(X[k].reshape(-1, 1)))
                increment = X[k].reshape(-1, 1).dot((self.change_y(y[k]) - yhat).T)  # The gradient
                self.W += alpha * increment
                # if not (i % 10000):
                #     print(i)
        elif strategy=="batch":
            for i in range(times):
                increment = numpy.zeros((self.feature, self.typenum))  # The gradient
                for j in range(self.sample):  # Calculate all samples
                    yhat = self.softmax_calculation(self.W.T.dot(X[j].reshape(-1, 1)))
                    increment += X[j].reshape(-1, 1).dot((self.change_y(y[j]) - yhat).T)
                # print(i)
                self.W += alpha / self.sample * increment
        else:
            raise Exception("Unknown strategy")
