"""
Reading in and loading nuclear data, such as the masses, binding energies etc.
"""

import numpy as np 
import pandas as pd 
import os 
import sys
import requests, bs4, re, time
from sklearn import preprocessing



class load_data():
    def __init__(self, dataset = "2016"):
        self.data_path = "../data/"
        self.result_path = "../results/"
        self.lifetime_filename = 'lifetimes_list'
        self.units = {'ms' : 10e-6, 's' : 1., 'm' : 60., 'h' : 60*60., 'd' : 24*60*60., 'y' : 365.24*24*60*60.}


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

        if read_rct1:
            infile_rct1 = open("../input/rct1_16.txt",'r')

            Rct1 = pd.read_fwf(infile_rct1, usecols=(1,3,4,6,8,10,14),
                names=('A', 'Z', 'S(2n)', 'S(2p)', 'Q(a)', 'Q(2B-)', 'Q(B-)'),
                widths=(1,3,3,4,11,8,10,8,10,8,10,8,10,8,10,8),
                header=39,
                index_col=False)

        if read_rct2:
            infile_rct2 = open("../input/rct2_16.txt",'r')

            Rct2 = pd.read_fwf(infile_rct2, usecols=(1,3,4,6,8,10,12,14),
                names=('A', 'Z', 'S(n)', 'S(p)', 'Q(4B-)', 'Q(d,a)', 'Q(p,a)', 'Q(n,a)'),
                widths=(1,3,3,4,11,8,10,8,10,8,10,8,10,8,10,8),
                header=39,
                index_col=False)

        
        if betarates:
            # File format:
            #A   Z   T9   log_10{rho(g/cm3) * Ye}   mu_e(electron chemical potential)  rates(beta+)  rate(EC) rates(nu) rates(beta-) rates(e+ capture)  rates(nubar)  tableID

            infile_betarates = open("../input/data_too_big_for_github/single_rate_table.txt",'r')

            Betarates = pd.read_fwf(infile_betarates, usecols=(0,1,2,3,6),
                names=('A', 'Z', 'T9', 'rho', 'EC' ),
                widths=(4,4,10,10,10,10,10,10),
                header=0,
                index_col=False)

        data = pd.merge(Masses, Rct1, on=["A", "Z"])   #, Rct1)
        data = pd.merge(data, Rct2, on=["A", "Z"])
        data = pd.merge(data, Betarates.loc[(Betarates['T9'] == 0.4) & (Betarates['rho']== 4.0)], on=["A", "Z"])


        if strip_of_invalid_values:
            data['Ebinding'] = pd.to_numeric(data['Ebinding'], errors='coerce')
            data['Atomic mass'] = pd.to_numeric(data['Atomic mass'], errors='coerce')
            data['S(n)'] = pd.to_numeric(data['S(2n)'], errors='coerce')
            data['S(2n)'] = pd.to_numeric(data['S(2n)'], errors='coerce')
            data['S(p)'] = pd.to_numeric(data['S(p)'], errors='coerce')
            data['S(2p)'] = pd.to_numeric(data['S(2p)'], errors='coerce')
            data['Q(a)'] = pd.to_numeric(data['Q(a)'], errors='coerce')
            data['Q(d,a)'] = pd.to_numeric(data['Q(d,a)'], errors='coerce')
            data['Q(p,a)'] = pd.to_numeric(data['Q(p,a)'], errors='coerce')
            data['Q(n,a)'] = pd.to_numeric(data['Q(n,a)'], errors='coerce')
            data['Q(B-)'] = pd.to_numeric(data['Q(B-)'], errors='coerce')
            data['EC'] = pd.to_numeric(data['EC'], errors='coerce')

        if drop_nan:    
            data = data.dropna()
        
        self.data = data
        print("Data loaded succsessfully. Well done :)")


    def scrape_internet(self, add_lifetime_to_df = True):
        n = 8000 # Arbitrary
        protonrange = range(4,98)
        lifetimes_list = [] 

        index = 0
        for protonnumber in protonrange:
            url = "http://nucleardata.nuclear.lu.se/toi/listnuc.asp?sql=&Z=%d" % protonnumber
            thread = requests.get(url)
            cleanthread = str(bs4.BeautifulSoup(thread.text, 'html.parser'))
            soup = cleanthread
            soup = soup.split('<th><a href=')
            for i in range(1,len(soup)):
                line = soup[i].split('\n') # Split all lines with different information
                if 'm' not in list(line[0])[30:34]: #Avoid isomers

                    Z = int(re.sub('</td>', '', re.sub('<td>', '', line[1])))
                    N = int(re.sub('</td>', '', re.sub('<td>', '', line[2])))

                    lifetime = line[4].split('<i>') # Extract lifetime from uncertainty number at the end.
                    # Remove a lot of html-markers
                    lifetime = re.sub('\xa0', '', lifetime[0])
                    lifetime = re.sub('<td>', '', lifetime)
                    lifetime = re.sub('&gt;', '', lifetime)
                    lifetime = re.sub('</td>', '', lifetime)
                    

                    if 'stable' in lifetime: # If stable nuclei
                        lifetimes_list.append([Z+N, Z, N, None])
                        print(Z, N, 'stable')

                    else:
                        try: # If a lifetime is given:
                            # Extract unit of lifetime. y/m/s/ms ...etc
                            lifetime_str_list = list(lifetime)
                            unit = re.findall("[a-zA-Z]+", lifetime_str_list[-3] + lifetime_str_list[-2] + lifetime_str_list[-1]) 
                            # Calculate the log10 of the (factor times the unit) to find lifetime in log10(sec)
                            lifetime_log = np.log10(float(re.sub(unit[0], '', lifetime)) * self.units[unit[0]])
                            print(' A: %3.f Z: %3.f N %3.f ' %(N+Z, Z, N), 'Lifetime: ', lifetime, 'Lifetime log(sec): ', lifetime_log)
                            # Add result to matrix
                            lifetimes_list.append([N+Z, Z, N, lifetime_log])
                            index += 1

                        except: # If no lifetime written down.
                            print(' A: %3.f Z: %3.f N %3.f ' %(N+Z, Z, N), line[4], 'is not a valid or excisting lifetime/number')

        lifetime_matrix = np.empty((len(lifetimes_list), 4))
        lifetime_matrix[:] = lifetimes_list
        np.save(self.lifetime_filename, lifetime_matrix)   
        self.Lifetimes = lifetimes_list 
        self.Lifetime_df = pd.DataFrame(lifetimes_list, columns=['A', 'Z', 'N', 'lifetime'])
        print("Lifetimes scraped off the internet. ")
        if add_lifetime_to_df:
            self.data = pd.merge(self.data, self.Lifetime_df, on=["A", "Z", "N"])
            print("Lifetimes merged into data.")

    def load_lifetimes(self, add_lifetime_to_df = True):
        if not os.path.exists(self.lifetime_filename+'.npy'):
            print("Path/file ", self.lifetime_filename+'.npy', "does not exist. Please run load_data().scrape_internet() in file load_nuclear_data.py to generate file first.")
            raise FileNotFoundError

        lifetimes_matrix = np.load(self.lifetime_filename+'.npy')
        self.Lifetime_df = pd.DataFrame(lifetimes_matrix, columns=['A', 'Z', 'N', 'lifetime'])
        print("Lifetimes loaded from memory. ")
        if add_lifetime_to_df:
            self.data = pd.merge(self.data, self.Lifetime_df, on=["A", "Z", "N"])
            print("Lifetimes merged into data.")
    
    def drop_unused_columns(self):
        self.data = self.data.drop(columns=['Q(2B-)', 'Q(4B-)', 'T9', 'rho'])

    def normalise_dataset(self):
        data = self.data
        for column in self.data:
            if column != 'Element':
                data[column]=(data[column]-data[column].mean())/data[column].std()

        self.data = data

if __name__ == "__main__":
    run = load_data()
    run.make_dir()
    run.read_data()
