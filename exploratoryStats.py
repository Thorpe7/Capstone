import numpy as np
import pandas as pd



def measure_stats(path_to_csv):
    '''
    Perfroms exploratory statistics on time series data

    Input
    --------------------
    path_to_csv: filepath to data (.csv)

    Output
    --------------------
    

    '''

    # Read in data
    data = pd.read_csv(path_to_csv)
    f = open('descriptive_statistics.txt', 'w')
    f.write("Total number of participants: " + str(len(data["ID"].unique())) + "\n")
    print(data.head())
    f.close()

    

if __name__ == "__main__":
    measure_stats('test_measure.csv')
    # subject_stats('subject-info.csv')
    