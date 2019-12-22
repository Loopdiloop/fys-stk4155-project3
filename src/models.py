import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

""" Models for fitting the data... """




class fit_models():
    def __init__(self, df, training_fraction = 0.8):
        
        self.df = df 

        mask = np.random.rand(len(df)) < training_fraction
        self.training = self.df[mask]
        self.training= self.training.dropna()
        self.trainingX = self.training.drop(columns=['lifetime', 'Element', 'A'])
        self.trainingy = self.training['lifetime']
        print(self.trainingX)

        self.test = self.df[~mask]
        self.test = self.test.dropna()
        self.testX = self.test.drop(columns=['lifetime', 'Element', 'A'])
        self.testy = self.test['lifetime']

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


    def random_forest_sklearn(self):

        testX = self.testX
        testy = self.testy

        X = self.trainingX
        y = self.trainingy
        
        #classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        regressor = RandomForestRegressor(n_estimators = 300, random_state = 2)
        regressor.fit(X, y)
        #MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2, random_state=1, solver='lbfgs')

        #testX = test.loc[:, test.columns !=  ['lifetime', 'Element', 'A']]
        #testY = np.array(test['lifetime'])
        
        prediction = regressor.predict(testX)
        testy_array = testy.to_numpy()
        print('TESTING RESULTS! ')

        xx = np.linspace(0,1,len(prediction))
        #testX['Z'].to_numpy()
        plt.plot(xx, testy_array, '*', label='actual value')
        plt.plot(xx, prediction,  '*', label='prediction')
        plt.plot(xx, abs(testy_array-prediction), '*', label='abs difference')
        plt.legend()
        plt.show()

        print('Avg. abs. diff: ', np.mean(abs(testy_array-prediction)))
        #for i in range(len(prediction)):
        #    print(testy[i],prediction[i])


