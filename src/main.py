""" Main program for running the nucelar lifetime models. """

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import time
import sys 

# Other classes and functions:
from load_nuclear_data import load_data
from models import fit_models
""" 
    Load data contains:
    __init__(self, dataset = "2016")
    def make_dir(self)
    def read_data(self)
"""

'''x = np.linspace(1e-20, 1e20, 500)
plt.plot(x, np.log(x), '*')
plt.xlabel('A')
plt.ylabel('liftime [log(s)]')
plt.show()'''

nuclear_data = load_data()
#nuclear_data.make_dir()
nuclear_data.read_data()

#nuclear_data.scrape_internet(add_lifetime_to_df = True)
nuclear_data.load_lifetimes()

'''plt.plot(nuclear_data.data['A'], nuclear_data.data['Q(a)'], '*')
plt.show()
'''



#print(nuclear_data.data)
nuclear_data.drop_unused_columns()
#print(nuclear_data.data)
nuclear_data.normalise_dataset()

model = fit_models(df = nuclear_data.data, training_fraction = 0.7)
model.random_forest_sklearn()





sys.exit()

df_lifetime = nuclear_data.Lifetime_df

"""
plt.semilogy(df_lifetime['A'], np.exp(df_lifetime['lifetime']), '*')
plt.xlabel('A')
plt.ylabel('liftime [log(s)]')
plt.show()
plt.clf()"""



'''
# Print for all Z = 75 nuclei
df_one_Z = df_lifetime[df_lifetime['Z']==75]
print(df_one_Z)

plt.plot(df_one_Z['A'], df_one_Z['lifetime'], '*')
plt.xlabel('A')
plt.ylabel('liftime [log(s)]')
plt.show()'''

# Lifetimes for Z/N ?? Z/(A-Z)
Z_N = df_lifetime['N']/(df_lifetime['Z'])
lifetimes_Z_N = df_lifetime[Z_N < 3]
Z_N = Z_N[Z_N < 3]
plt.plot(Z_N, lifetimes_Z_N['lifetime'], '*')
plt.xlabel('N/Z')
plt.ylabel('liftime [log(s)]')
plt.show()
plt.clf()

'''
# Plot N vs. Z of stable nuclei (should be no surprise...)
stable_nuclei = df_lifetime[df_lifetime['lifetime'] == None]
plt.plot(stable_nuclei['N'], stable_nuclei['Z'], '*')
plt.xlabel('N')
plt.ylabel('Z')
plt.show()
plt.clf()
'''



#print(nuclear_data.data)


#Den svarte viking

