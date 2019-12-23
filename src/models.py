import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import plotting
import sys

""" Models for fitting the data... """


class fit_models():
    def __init__(self, df, training_fraction = 0.8):
        
        self.df = df 

        mask = np.random.rand(len(df)) < training_fraction
        self.training = self.df[mask]
        self.test = self.df[~mask]


        self.training_unstable= self.training.dropna()
        self.trainingX = self.training_unstable.drop(columns=['lifetime', 'Element', 'A'])
        self.trainingy = self.training_unstable['lifetime']

        self.test_unstable = self.test.dropna()
        self.testX = self.test_unstable.drop(columns=['lifetime', 'Element', 'A'])
        self.testy = self.test_unstable['lifetime']
        self.testy_np = self.testy.to_numpy()


        # Binary testing data for stable/ not stable classification
        self.training_stability = self.training
        self.trainingX_st = self.training_stability.drop(columns=['lifetime', 'Element', 'A'])
        self.trainingy_st = self.training_stability['lifetime'].fillna(0)
        self.trainingy_st[self.trainingy_st != 0] = 1
        self.trainingy_st = self.trainingy_st.astype(int)

        self.test_stability = self.test
        self.testX_st = self.test_stability.drop(columns=['lifetime', 'Element', 'A'])
        self.testy_st = self.test_stability['lifetime'].fillna(0)
        self.testy_st[self.testy_st != 0] = 1
        self.testy_st = self.testy_st.astype(int)
        self.testy_st_np = self.testy_st.to_numpy()

    def fit_logistic_regression_sklearn(self, delta=0.01, iterations = 1000):
        """ Linear, logistic fit. Try to classify stable/non-stable nuclei."""
        
        testX = self.testX_st
        testy = self.testy_st

        trainX = self.trainingX_st
        trainy = self.trainingy_st
        
        # n = no. of users, p = predictors, i.e. parameters.
        n,p = trainX.shape

        X = np.ones((n,p+1))
        X[:,1:] = trainX
        X[:,0] = 1

        y = trainy

        skl_reg = LogisticRegression(solver='lbfgs') #solver="lbfgs")
        skl_reg.fit(X,y)

        n,p = testX.shape
        X_test = np.ones((n,p+1))
        X_test[:,1:] = testX
        X_test[:,0] = 1

        self.logistic_prediction = skl_reg.score(X_test, testy)
        

    def random_forest_sklearn(self):

        testX = self.testX
        testy = self.testy

        X = self.trainingX
        y = self.trainingy
        
        regressor = RandomForestRegressor(n_estimators = 300, random_state = 2)
        regressor.fit(X, y)

        self.forest_prediction = regressor.predict(testX)
        
        
        testy_array = testy.to_numpy()
        #plotting.plot_predictions_data(self.forest_prediction, testy_array)

