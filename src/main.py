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
#nuclear_data.make_dir()
#nuclear_data.read_data()

'''plt.plot(nuclear_data.data['A'], nuclear_data.data['Q(a)'], '*')
plt.show()
'''

nuclear_data.scrape_internet()


#print(nuclear_data.data)
