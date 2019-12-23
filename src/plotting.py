
import matplotlib.pyplot as plt 
import numpy as np

def plot_predictions_data(prediction, testy):
    xx = np.linspace(0,1,len(prediction))
    #testX['Z'].to_numpy()
    plt.plot(xx, testy, '*', label='actual value')
    plt.plot(xx, prediction,  '*', label='prediction')
    plt.plot(xx, abs(testy-prediction), '*', label='abs difference')
    plt.legend()
    plt.title('Predictions made by random forest from sklearn.')
    plt.show()
    plt.savefig('../results/figures/prediction_random_forest.png')
    plt.clf()

def plot_A_lifetimes(df):
    plt.plot(df['A'], df['lifetime'], '*')
    plt.xlabel('A')
    plt.ylabel('liftime [log(s)]')
    plt.show()
    plt.clf()


def plot_for_Z(df, Z_value):
    # Print for all Z = Z_value nuclei
    df_one_Z = df[df['Z']==Z_value]
    plt.plot(df_one_Z['A'], df_one_Z['lifetime'], '*')
    plt.xlabel('A')
    plt.ylabel('liftime [log(s)]')
    plt.show()
    plt.clf()

def plot_for_N_Z(df, Z_value):
    # Lifetimes for Z/N ?? Z/(A-Z)
    Z_N = df['N']/(df['Z'])
    lifetimes_Z_N = df[Z_N < 3]
    Z_N = Z_N[Z_N < 3]
    plt.plot(Z_N, lifetimes_Z_N['lifetime'], '*')
    plt.xlabel('N/Z')
    plt.ylabel('liftime [log(s)]')
    plt.show()
    plt.clf()

def plot_stable(df):
    # Plot N vs. Z of stable nuclei (should be no surprise...)
    stable_nuclei = df[df['lifetime'] == None]
    plt.plot(stable_nuclei['N'], stable_nuclei['Z'], '*')
    plt.xlabel('N')
    plt.ylabel('Z')
    plt.show()
    plt.clf()
