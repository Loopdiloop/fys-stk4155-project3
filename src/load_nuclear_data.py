"""
Reading in and loading nuclear data, such as the masses, binding energies etc.
"""

import numpy as np 
import pandas as pd 
import os 
import sys

import requests, bs4, re, time

class load_data():
    def __init__(self, dataset = "2016"):
        self.data_path = "../data/"
        self.result_path = "../results/"

        self.units = {'ms' : 10e-6, 's' : 1., 'm' : 60., 'h' : 60*60., 'd' : 24*60*60., 'y' : 365.24*24*60*60.}

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


        """data = pd.merge(Masses,
                 Rct1[['S(2n)' 'S(2n)', 'S(2p)', 'Q(a)', 'Q(2B-)', 'Q(B-)']],
                 on='use_id')"""

        
        if betarates:
            # File format:
            #A   Z   T9   log_10{rho(g/cm3) * Ye}   mu_e(electron chemical potential)  rates(beta+)  rate(EC) rates(nu) rates(beta-) rates(e+ capture)  rates(nubar)  tableID
            
            #infile_betarates = open("../input/data_too_big_for_github/for_each_nucl/20_9.txt",'r')
            infile_betarates = open("../input/data_too_big_for_github/single_rate_table.txt",'r')

            '''Betarates = pd.read_fwf(infile_betarates, usecols=(0,1,2),#,3,5,6,8),
                names=('A', 'Z', 'T9', 'rho', 'beta+', 'EC', 'beta-',  ),
                widths=(4,4,10,10,10,10,10,10),
                header=0,
                index_col=False)'''


            Betarates = pd.read_fwf(infile_betarates, usecols=(0,1,2,3,6),
                names=('A', 'Z', 'T9', 'rho', 'EC' ),
                widths=(4,4,10,10,10,10,10,10),
                header=0,
                index_col=False)
        #   6   3     0.100     5.000     0.052       ---  -100.000  -100.000       ---       ---       ---      0
        #data = pd.concat([Masses, Rct1, Rct2], axis=1)
        print(Betarates)
        print(Betarates.loc[(Betarates['T9'] == 0.4) & (Betarates['rho']== 4.0)])
        data = pd.merge(Masses, Rct1, on=["A", "Z"])   #, Rct1)
        data = pd.merge(data, Rct2, on=["A", "Z"])
        data = pd.merge(data, Betarates.loc[(Betarates['T9'] == 0.4) & (Betarates['rho']== 4.0)], on=["A", "Z"])
        #data = data.groupby('A')
        #print(data)


        if strip_of_invalid_values:
            data['Ebinding'] = pd.to_numeric(data['Ebinding'], errors='coerce')
            data['S(n)'] = pd.to_numeric(data['S(2n)'], errors='coerce')
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
        print(data.size)


        self.data = data
        print("Data loaded succsessfully. Well done :)")







    def scrape_thread(cleanthread):		# We need to feed the thread data into the function
        singlethreadlinksearch = re.compile(r'\<a class="storylink" href="(.+?)"\>')		
        singlethreadlink = singlethreadlinksearch.findall(str(cleanthread))
        commenterIDsearch = re.compile(r'user\?id=(.+?)"')
        commenterIDs = commenterIDsearch.findall(str(cleanthread))
        try:
            firstcommenter = commenterIDs[1]		# If there are no commenters this will fail, so we wrap it in a try/except just in case
        except:
            firstcommenter = "No commenters"
        return singlethreadlink, firstcommenter		# Return the variables





    def scrape_internet(self):
        n = 8000
        # Z, N, lifetime value in log(sec) OR if stable == 20.0
        stable = 20.0
        matrix_lifetimes = np.empty((n,3))

        index = 0
        for massnumber in range(74,76):
            url = "http://nucleardata.nuclear.lu.se/toi/listnuc.asp?sql=&Z=%d" % massnumber
            #url = 'https://www.google.com/'
            thread = requests.get(url)
            cleanthread = str(bs4.BeautifulSoup(thread.text, 'html.parser'))
            # First nuclei at line 51.
            soup = cleanthread#.prettify()
            soup = soup.split('<th><a href=')
            for i in range(1,len(soup)):
                print(i, soup[i].split('\n'))#.children) #.split('\n'))
                line = soup[i].split('\n')
                if 'm' not in list(line[0])[30:34]: #Avoid isomers
                    print(list(line[0])[30:34])
                    #Z:
                    #Z = int(list(line[1])[4] + list(line[1])[5])
                    #Z = re.sub('<td>', '', line[1] )#line[1].strip('<td>', '</td>')
                    Z = int(re.sub('</td>', '', re.sub('<td>', '', line[1])))
                    # N
                    N = int(re.sub('</td>', '', re.sub('<td>', '', line[2])))
                    #N = int(list(line[2])[4] + list(line[2])[5])
                    print('    Z', Z, '    N', N)

                    lifetime = line[4].split('<i>')
                    lifetime = re.sub('\xa0', '', lifetime[0])
                    lifetime = re.sub('<td>', '', lifetime)
                    lifetime = re.sub('&gt;', '', lifetime)
                    lifetime = re.sub('</td>', '', lifetime)
                    print('LIFETIME CLEAN ?? ', lifetime)
                    lifetime_list = list(lifetime)
                    if 'stable' in lifetime:
                        matrix_lifetimes[index] = np.array([Z, N, stable])
                        index += 1
                    else:
                        try:
                            unit = re.findall("[a-zA-Z]+", lifetime_list[-2] + lifetime_list[-1]) # Extract unit of lifetime. y/m/s/ms ...etc
                            lifetime_log = np.log(float(re.sub(unit[0], '', lifetime)) * self.units[unit[0]])
                            
                            print('Yoooo', lifetime, lifetime_log)

                            print(lifetime, unit)
                            print(self.units[unit[0]]) #Find corresponding log in seconds.
                            print(lifetime)
                            matrix_lifetimes[index] = np.array([Z, N, lifetime_log])
                            index += 1

                        except:
                            print(lifetime, 'is not a valid or excisting lifetime/number')
                    
                    # After </A></TH> : Z, N
                    # <TH ><A HREF=nuclide.asp?iZA=740158><SUP>158</SUP>W</A></TH>
                    #Z = 34
                    #N = 34
                    #print(r.text)

                    # Fill matrix

                    
        print(matrix_lifetimes[:50])            
        #df_lifetimes = pd.DataFrame(matrix_lifetimes, columns=['Z', 'N', 'lifetime'])







if __name__ == "__main__":
    run = load_data()
    run.make_dir()
    run.read_data()
