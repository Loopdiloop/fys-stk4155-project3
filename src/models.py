import numpy as np
import pandas as pd 


""" Models for fitting the data... """




class fit_models():
    def __init__(self, df):
        moddd = True

        self.df = df 
        



    def fit_logistic_regression_sklearn(self, delta=0.01, iterations = 1000):
        """ Do a linear, logistic fit for matrix X with sigmoid function, from sklearn"""
        
        # n = no. of users, p = predictors, i.e. parameters.
        n,p = np.shape(self.inst.XTrain)

        X = np.ones((n,p+1))
        X[:,1:] = self.inst.XTrain
        X[:,0] = 1

        y = self.inst.yTrain

        skl_reg = LogisticRegression(solver='lbfgs') #solver="lbfgs")
        skl_reg.fit(X,y)

        n,p = np.shape(self.inst.XTest)
        X_test = np.ones((n,p+1))
        X_test[:,1:] = self.inst.XTest
        y_test = self.inst.yTest
        X_test[:,0] = 1

        score = skl_reg.score(X_test, self.inst.yTest)
        print("SKLEARN SCORE: ", score)