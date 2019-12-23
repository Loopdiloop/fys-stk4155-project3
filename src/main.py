""" Main program for running the nucelar lifetime models. """

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import time
import sys 

# Other classes and functions:
from load_nuclear_data import load_data
from models import fit_models
from statistics import calc_statistics

no_tests = 12 #number of different fits to be done.

nuclear_data = load_data()
# If directories does not exist:
#nuclear_data.make_dir()

# Read data from files locally. 
# Another file that is too big for github is also needed! See documentation.
nuclear_data.read_data()

# Scrape lifetimes from the internet and save as a .npy file
#nuclear_data.scrape_internet(add_lifetime_to_df = True)
# Or load already scraped data
nuclear_data.load_lifetimes()

# Drop some unusful columns we are not currently using.
nuclear_data.drop_unused_columns()

# Normalise columns of the data for easier processing and modelling.
nuclear_data.normalise_dataset()

for i in range(no_tests):
    # Initialise the class with the models with our data.
    model = fit_models(df = nuclear_data.data, training_fraction = 0.8)
    # Run a random forest for classifying binary stable/not stable nuclei.

    model.random_forest_sklearn()
    # Try to predict lifetime of unstable nuclei.
    model.fit_logistic_regression_sklearn()

    print("\n Classifying stable/ unstable nuclei: \n")
    print("Logistic score: ", model.logistic_prediction)

    print("\n Predicting lifetimes: \n")
    calc_statistics(model.testy_np, model.forest_prediction)

print('Fin')

