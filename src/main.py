""" Main program for running the nucelar lifetime models. """

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Other classes and functions:
from load_nuclear_data import load_data
""" 
    Load data contains:
    __init__(self, dataset = "2016")
    def make_dir(self)
    def read_data(self)
"""


nuclear_data = load_data()
nuclear_data.make_dir()
nuclear_data.read_data()

print(nuclear_data.data)
