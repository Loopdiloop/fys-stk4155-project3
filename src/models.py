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

        self.training_unstable= self.training.dropna()
        self.trainingX = self.training_unstable.drop(columns=['lifetime', 'Element', 'A'])
        self.trainingy = self.training_unstable['lifetime']
        print(self.trainingX)

        self.test = self.df[~mask]

        self.test_stability = self.test#.drop(columns=['lifetime', 'Element', 'A'])
        self.testX_st = self.test_stability.drop(columns=['lifetime', 'Element', 'A'])
        self.testy_st = self.test_stability['lifetime'].fillna(0)
        #pd.to_numeric(self.test_stability['lifetime'], errors=0)
        self.testy_st[self.testy_st != 0] = 1
        self.testy_st = self.testy_st.astype(int)
        #df.a = df.a.astype(float)

        #self.testy_st[type(self.testy_st[:]) == float] = 1
        #self.testy_st = self.test_st.fillna()
        print(self.testy_st)
        sys.exit()

        self.test_unstable = self.test.dropna()
        self.testX = self.test_unstable.drop(columns=['lifetime', 'Element', 'A'])
        self.testy = self.test_unstable['lifetime']

    def fit_logistic_regression_sklearn(self, delta=0.01, iterations = 1000):
        """ Linear, logistic fit. Try to classify stable/non-stable nuclei."""
        
        testX = self.testX_stability
        testy = self.testy_stability

        trainX = self.trainingX_
        trainy = self.trainingy
        
        # n = no. of users, p = predictors, i.e. parameters.
        n,p = trainX.shape
        print(trainX.shape)

        X = np.ones((n,p+1))
        X[:,1:] = trainX
        X[:,0] = 1

        y = trainy

        skl_reg = LogisticRegression(solver='lbfgs') #solver="lbfgs")
        skl_reg.fit(X,y)

        n,p = testX.shape
        testX = np.ones((n,p+1))
        testX[:,1:] = testX
        testX[:,0] = 1

        score = skl_reg.score(testX, testy)
        print("SKLEARN SCORE: ", score)


    def random_forest_sklearn(self):

        testX = self.testX
        testy = self.testy

        X = self.trainingX
        y = self.trainingy
        
        regressor = RandomForestRegressor(n_estimators = 300, random_state = 2)
        regressor.fit(X, y)

        prediction = regressor.predict(testX)
        testy_array = testy.to_numpy()

        plotting.plot_predictions_data(prediction, testy_array)

        print('Avg. abs. diff: ', np.mean(abs(testy_array-prediction)))
