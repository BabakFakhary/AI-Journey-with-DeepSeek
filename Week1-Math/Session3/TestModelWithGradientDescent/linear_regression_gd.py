#                                                                               به نام خدا
# install: pip install --upgrade arabic-reshaper
import arabic_reshaper
#-------------------------------------------------------
# install: pip install python-bidi
from bidi.algorithm import get_display
#-------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.cost_history = []
    
    def fit(self, X, y):
        # اضافه کردن ستون 1 برای ضریب اریب
        X_b = np.c_[np.ones((len(X), 1)), X]
        
        # مقداردهی اولیه پارامترها عددی بین صفر و یک
        self.theta = np.random.randn(X_b.shape[1], 1)
        
        # آموزش مدل
        for iteration in range(self.n_iterations):
            predictions = X_b.dot(self.theta)
            errors = predictions - y
            gradients = 2/len(X_b) * X_b.T.dot(errors)
            self.theta -= self.learning_rate * gradients
            self.cost_history.append((errors**2).mean())
    
    def predict(self, X):
        X_b = np.c_[np.ones((len(X), 1)), X]
        return X_b.dot(self.theta)
    
    def plot_cost_history(self):
        plt.plot(range(self.n_iterations), self.cost_history)
        plt.xlabel(get_display(arabic_reshaper.reshape('تعداد تکرارها')))
        plt.ylabel(get_display(arabic_reshaper.reshape('تابع هزینه')))
        plt.title(get_display(arabic_reshaper.reshape('تاریخچه تابع هزینه')))
        plt.show()