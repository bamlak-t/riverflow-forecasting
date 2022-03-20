import numpy as np
import pandas as pd
 

class DataSet:
    def __init__(self, filename, limit):
        
        # parse excel with 'none' values for 'a' and '#'
        dataset = pd.read_excel(filename+".xlsx", na_values=["a", "#"], dtype=float, parse_dates=['Date']) 

        # convert date into day number (1-365/366) of the year and delete Date column
        dataset["DateDays"] = [float(x.strftime('%j')) for x in dataset["Date"]]
        del dataset["Date"]

        # filter values greater than 3 stds from the mean and negative values
        dataset = dataset[ (dataset < dataset.mean() + ( 3*dataset.std() )) & (dataset >= 0) ].dropna()

        # standardise data based on a min and max value
        dataset = ( (dataset-dataset.min())/(dataset.max()-dataset.min()) ) * (limit["max"] - limit["min"]) + limit["min"]

        # randomly shuffle rows
        dataset = dataset.sample(frac=1)


        self.data = dataset

    def getDataset(self, columns = []):
        return self.data if columns == [] else self.data[columns]





d = DataSet(filename = "DataSet", limit = {"min": 0.1, "max": 0.9})
print(d.getDataset(["Skelton"]))


