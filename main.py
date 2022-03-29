from typing import final
from matplotlib import markers
import numpy as np
import pandas as pd
from ModelClass import Model
import matplotlib.pyplot as plt
import time


import cProfile, pstats, io
from pstats import SortKey

class DataSet:
	def __init__(self, filename: str, limit: dict, lagby: int) -> None:
		# parse excel with 'none' values for 'a' and '#'
		dataset = pd.read_excel(filename+".xlsx", na_values=["a", "#"], dtype=float, parse_dates=['Date']) 

		# dataset = dataset.head(11)

		# convert date into day number (1-365/366) of the year
		dataset["Date"] = [float(x.strftime('%j')) for x in dataset["Date"]]

		# lag dataset
		dataset["Skelton"] = dataset["Skelton"].shift(periods=lagby)

		# filter values greater than 3 stds from the mean and negative values
		dataset = dataset[ (dataset < dataset.mean() + ( 3*dataset.std() )) & (dataset >= 0) ].interpolate().dropna() 

		self.outputMax = dataset["Skelton"].max()
		self.outputMin = dataset["Skelton"].min()
		self.limit = limit

		# standardise data based on a min and max value
		dataset = ( (dataset-dataset.min())/(dataset.max()-dataset.min()) ) * (limit["max"] - limit["min"]) + limit["min"]

		# randomise dataset to mitigate data missrepresentation
		dataset = dataset.sample(frac=1)

		print(dataset, len(dataset))

		self.data = dataset

	def getDataset(self, columns: list = [], split: list = [0.6, 0.2, 0.2]) -> list:

		trainLen = split[0]
		validateLen = split[1]

		returnSet = self.data.copy()

		returnSet = returnSet if columns == [] else returnSet[columns]

		trainingData, validationData, testingData = np.split( 
														returnSet.sample(frac=1, random_state=1).to_numpy(), 
														[int(trainLen*len(returnSet)), int((trainLen+validateLen)*len(returnSet))]
													)

		return trainingData, validationData, testingData

	def standardise(self, value: float) -> float:
		return ( (value-self.outputMin)/(self.outputMax-self.outputMin) ) * (self.limit["max"] - self.limit["min"]) + self.limit["min"]

	def inverseStandardise(self, standard: float) -> float:
		return ((( standard - self.limit["min"] ) / (self.limit["max"] - self.limit["min"])) * (self.outputMax - self.outputMin) ) + self.outputMin		

def outSet(set, num):
	for i, val in enumerate(set):
		print(val)
		if (i+1 == num):
			break

d = DataSet(
        filename = "Dataset", 
        limit = {"min": 0, "max": 1},
        lagby=1
    )

trainingData, validationData, testingData = d.getDataset(["Date", "Crakehill", "Skip Bridge", "Westwick", "Snaizeholme", "Arkengarthdale", "East Cowton", "Malham Tarn", "Skelton"])

print("\nTRAINING DATA", len(trainingData))
outSet(trainingData, 6)
print("\nValidation DATA", len(validationData))
outSet(validationData, 5)
print("\nTESTING DATA", len(testingData))
outSet(testingData, 5)

m = Model(
        network = {
                "inputs": 8,
                "hiddenLayers": [8],
                "outputs": 1
        },
        activation = "sigmoid"
    )

print("BEFORE TRAINING")

print("TRAINING ERROR")
btr = m.test(dataset = trainingData, inverseStandardise = d.inverseStandardise)
print( btr)

print("TESTING ERROR")
bte = m.test(dataset = testingData, inverseStandardise = d.inverseStandardise)
print( bte)

print("---------- AFTER TRAINING ----------")

startTime = time.time()
pr = cProfile.Profile()

pr.enable()

trainingInfo, validationInfo, epocs = m.train(
    trainingParams={
        "epocs": 1000,
        "batch_size": 2,
        "learning_rate": [0.1,0.05],
        "validation_error_check_freq": 100,
        "validation_error_increase_limit": 1.05
    },
    trainingData= trainingData,
    validationData=validationData,
    inverseStandardise = d.inverseStandardise
)
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())

print("TRAINING ERROR")
atr = m.test(dataset = trainingData, inverseStandardise = d.inverseStandardise)
print( atr)

print("TESTING ERROR")
ate = m.test(dataset = testingData, inverseStandardise = d.inverseStandardise)
print( ate) 

print("TOTAL TIME TAKEN:", time.time() - startTime, "SECS, FOR", epocs, "EPOCS")


# plot expected vs actual

trainingData, validationData, testingData = d.getDataset(["Date", "Crakehill", "Skip Bridge", "Westwick", "Snaizeholme", "Arkengarthdale", "East Cowton", "Malham Tarn", "Skelton"])
finalPlot = np.vstack([trainingData, validationData, testingData])
finalActual = np.empty((len(finalPlot),))
for i, row in enumerate (finalPlot):
	correctOutput = row[-1]
	inputs = row[:-1]
	actualOutput = m.forwardProp(inputs)[-1][1]
	finalActual[i] = actualOutput

# print(finalActual)
# print(finalPlot[:,-1])

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle("Error for training and validation sets")

ax1.scatter(finalActual, finalPlot[:,-1])
ax2.plot(trainingInfo[trainingInfo != 0], label="Training")
ax2.plot(validationInfo[validationInfo != 0], label="Validation")

ax2.set(xlabel='epocs', ylabel='error')


ax2.legend(loc='best')

plt.show()