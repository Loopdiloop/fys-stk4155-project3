""" Main program for running the nucelar lifetime models. """

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import time
import sys 

# Other classes and functions:
from load_nuclear_data import load_data
from models import fit_models



nuclear_data = load_data()
#nuclear_data.make_dir()
nuclear_data.read_data()

#nuclear_data.scrape_internet(add_lifetime_to_df = True)
nuclear_data.load_lifetimes()


nuclear_data.drop_unused_columns()

nuclear_data.normalise_dataset()

model = fit_models(df = nuclear_data.data, training_fraction = 0.8)
model.random_forest_sklearn()
model.fit_logistic_regression_sklearn()


