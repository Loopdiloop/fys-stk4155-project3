"""
Reading in and loading nuclear data, such as the masses, binding energies etc.
"""

import numpy as np 
import pandas as pd 
import os 
import sys

class load_data():
    def __init__(self, dataset = "2016"):
        self.data_path = "../data/"
        self.result_path = "../results/"

        """
        if dataset == "2016":
            self.infile = open("../input/masses_16.txt",'r')
        elif dataset == "2012":
            self.infile = open("../input/masses_12.txt",'r')
        else: 
            print("Error! Wrong input file, dataset = %s ?? Needs to be 2016 or 2012 or nothing." % dataset)"""

        ok = True
        print(" Is it ok? -", ok)


    def make_dir(self):
        result_path = self.result_path
        self.figure_path = "%s/figures"%result_path
        self.result_file_path = "%s/files/"%result_path

        if not os.path.exists(result_path):
            os.mkdir(result_path)

        if not os.path.exists(self.figure_path):
            os.makedirs(self.figure_path)

        if not os.path.exists(self.result_file_path):
            os.makedirs(self.result_file_path)

    def read_data(self):
        read_mass16 = True
        read_rct1 = True
        read_rct2 = True
        betarates = True

        strip_of_invalid_values = True
        drop_nan = True

        # Read the experimental data with Pandas from self.infile.
        if read_mass16:
            infile_mass = open("../input/masses_16.txt",'r')

            Masses = pd.read_fwf(infile_mass, usecols=(2,3,4,6,11,20),
                names=('N', 'Z', 'A', 'Element', 'Ebinding', 'Atomic mass'),
                widths=(1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11,1),
                header=39,
                index_col=False)

            # Extrapolated values are indicated by '#' in place of the decimal place, so
            # the Ebinding column won't be numeric. Coerce to float and drop these entries.
            
            #Masses['Ebinding'] = pd.to_numeric(Masses['Ebinding'], errors='coerce')
            #Masses = Masses.dropna()

            # Convert from keV to MeV.
            #Masses['Ebinding'] /= 1000

            # Group the DataFrame by nucleon number, A.
            #Masses = Masses.groupby('A')

            # Find the rows of the grouped DataFrame with the maximum binding energy.
            #Masses = Masses.apply(lambda t: t[t.Ebinding==t.Ebinding.max()])

        if read_rct1:
            infile_rct1 = open("../input/rct1_16.txt",'r')

            Rct1 = pd.read_fwf(infile_rct1, usecols=(1,3,4,6,8,10,14),
                names=('A', 'Z', 'S(2n)', 'S(2p)', 'Q(a)', 'Q(2B-)', 'Q(B-)'),
                widths=(1,3,3,4,11,8,10,8,10,8,10,8,10,8,10,8),
                header=39,
                index_col=False)

            # Extrapolated values are indicated by '#' in place of the decimal place, so
            # the Ebinding column won't be numeric. Coerce to float and drop these entries.
            """Rct1['S(2n)'] = pd.to_numeric(Rct1['S(2n)'], errors='coerce')

            Rct1['S(2p)'] = pd.to_numeric(Rct1['S(2p)'], errors='coerce')

            Rct1['Q(a)'] = pd.to_numeric(Rct1['Q(a)'], errors='coerce')
            
            Rct1['Q(2B-)'] = pd.to_numeric(Rct1['Q(2B-)'], errors='coerce')

            Rct1['Q(B-)'] = pd.to_numeric(Rct1['Q(B-)'], errors='coerce')

            Rct1 = Rct1.dropna()"""

            # Convert from keV to MeV.
            #Rct1['Ebinding'] /= 1000

            # Group the DataFrame by nucleon number, A.
            #Rct1 = Rct1.groupby('A')

            # Find the rows of the grouped DataFrame with the maximum binding energy.
            #Rct1 = Rct1.apply(lambda t: t[t.Ebinding==t.Ebinding.max()])
            
            #print(Rct1)


        if read_rct2:
            infile_rct2 = open("../input/rct2_16.txt",'r')

            Rct2 = pd.read_fwf(infile_rct2, usecols=(1,3,4,6,8,10,12,14),
                names=('A', 'Z', 'S(n)', 'S(p)', 'Q(4B-)', 'Q(d,a)', 'Q(p,a)', 'Q(n,a)'),
                widths=(1,3,3,4,11,8,10,8,10,8,10,8,10,8,10,8),
                header=39,
                index_col=False)


        """data = pd.merge(Masses,
                 Rct1[['S(2n)' 'S(2n)', 'S(2p)', 'Q(a)', 'Q(2B-)', 'Q(B-)']],
                 on='use_id')"""

        
        if betarates:
            # File format:
            #A   Z   T9   log_10{rho(g/cm3) * Ye}   mu_e(electron chemical potential)  rates(beta+)  rate(EC) rates(nu) rates(beta-) rates(e+ capture)  rates(nubar)  tableID
            infile_betarates = open("../input/data_too_big_for_github/for_each_nucl/20_9.txt",'r')

            Betarates = pd.read_fwf(infile_betarates, usecols=(0,1,2,3,5,6,8),
                names=('A', 'Z', 'T9', 'rho', 'beta+', 'EC', 'beta-',  ),
                widths=(4,4,10,10,10,10,10,10),
                header=0,
                index_col=False)
        #   6   3     0.100     5.000     0.052       ---  -100.000  -100.000       ---       ---       ---      0
        #data = pd.concat([Masses, Rct1, Rct2], axis=1)

        print(Betarates)
        data = pd.merge(Masses, Rct1, on=["A", "Z"])   #, Rct1)
        data = pd.merge(data, Rct2, on=["A", "Z"])
        data = pd.merge(data, Betarates.loc[(Betarates['T9'] == 0.4) & (Betarates['rho']== 4.0)], on=["A", "Z"])
        #data = data.groupby('A')
        #print(data)


        """
        # Read the experimental data with Pandas from self.infile.

        Masses = pd.read_fwf(self.infile, usecols=(2,3,4,6,11),
            names=('N', 'Z', 'A', 'Element', 'Ebinding'),
            widths=(1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11,1),
            header=39,
            index_col=False)

        # Extrapolated values are indicated by '#' in place of the decimal place, so
        # the Ebinding column won't be numeric. Coerce to float and drop these entries.
        Masses['Ebinding'] = pd.to_numeric(Masses['Ebinding'], errors='coerce')
        Masses = Masses.dropna()

        # Convert from keV to MeV.
        Masses['Ebinding'] /= 1000

        # Group the DataFrame by nucleon number, A.
        Masses = Masses.groupby('A')

        # Find the rows of the grouped DataFrame with the maximum binding energy.
        Masses = Masses.apply(lambda t: t[t.Ebinding==t.Ebinding.max()])"""

        """
        drop_modelled = ['Ebinding', 'S(2n)', 'S(2p)','Q(a)','Q(2B-)','Q(B-)']

        for drops in drop_modelled:
            data[drops] = pd.to_numeric(DeprecationWarning[drops], errors='coerce')
        """
        if strip_of_invalid_values:
            data['Ebinding'] = pd.to_numeric(data['Ebinding'], errors='coerce')
            data['S(2n)'] = pd.to_numeric(data['S(2n)'], errors='coerce')
            data['S(2p)'] = pd.to_numeric(data['S(2p)'], errors='coerce')
            data['Q(a)'] = pd.to_numeric(data['Q(a)'], errors='coerce')
            #data['Q(2B-)'] = pd.to_numeric(data['Q(2B-)'], errors='coerce')
            #data['Q(B-)'] = pd.to_numeric(data['Q(B-)'], errors='coerce')
        if drop_nan:    
            data = data.dropna()
        


        print("********************* Test for Re:  *******************************")
        print(data[data['Element'] == 'Re'])

        print(" All: ")
        print(data)


        self.data = data
        print("Data loaded succsessfully. Well done :)")


if __name__ == "__main__":
    run = load_data()
    run.make_dir()
    run.read_data()
