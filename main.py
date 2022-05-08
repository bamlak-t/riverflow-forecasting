import numpy as np
import pandas as pd
from ModelClass import Model
import matplotlib.pyplot as plt
import time

class DataSet:
	def __init__(self, filename: str, limit: dict, lagby: dict, moveAvg: int, dateType: str, stdOut: int) -> None:

		# parse excel with 'none' values for 'a' and '#'
		dataset = pd.read_excel(filename+".xlsx", na_values=["a", "#"], dtype=float, parse_dates=['Date']) 

		# convert date into day number (1-365/366), month number (1-12) or week number (1-53)
		dataset["Date"] = [float(x.strftime('%'+dateType)) for x in dataset["Date"]]

		# use previous Skelton river-flow as a predictor for the present
		dataset["Lagged Skelton"] = dataset["Skelton"]

		# add moving average columns
		headers = ["Crakehill", "Skip Bridge", "Westwick", "Snaizeholmethe", "Arkengarthdale", "East Cowton", "Malham Tarn", "Skelton"]
		for header in headers:
			dataset[header+" Avg"] = dataset[header].rolling(window=moveAvg).mean()

		# lag dataset by specified number
		for key, value in lagby.items():
			dataset[key] = dataset[key].shift(periods=value)
		
		# filter values greater than 3 stds from the mean and negative values
		dataset = dataset[ (dataset < dataset.mean() + ( stdOut*dataset.std() )) & (dataset >= 0) ].interpolate().dropna() 

		# used for inverse standardisation function
		self.outputMax = dataset["Skelton"].max()
		self.outputMin = dataset["Skelton"].min()
		self.limit = limit

		# standardise data based on a min and max value
		dataset = ( (dataset-dataset.min())/(dataset.max()-dataset.min()) ) * (limit["max"] - limit["min"]) + limit["min"]

		# randomise dataset to mitigate data missrepresentation
		dataset = dataset.sample(frac=1)

		self.data = dataset

	# used to return a section of the dataset with specified columns
	def getDataset(self, columns: list = [], split: list = [0.6, 0.2, 0.2]) -> list:
		# training and validation percentages
		trainLen = split[0]
		validateLen = split[1]

		# used to specify certain columns from dataset
		returnSet = self.data.copy()
		returnSet = returnSet if columns == [] else returnSet[columns]

		# split data into training, validation and testing according to the split specified
		trainingData, validationData, testingData = np.split( 
														returnSet.sample(frac=1, random_state=1).to_numpy(), 
														[int(trainLen*len(returnSet)), int((trainLen+validateLen)*len(returnSet))]
													)
		return trainingData, validationData, testingData

	# save dataset based for given name
	def saveDataset(self, name: str) -> None:
		self.data.to_excel(name+".xlsx")

	# used to standardise a value
	def standardise(self, value: float) -> float:
		return ( (value-self.outputMin)/(self.outputMax-self.outputMin) ) * (self.limit["max"] - self.limit["min"]) + self.limit["min"]

	# used to inverse standardise a value
	def inverseStandardise(self, standard: float) -> float:
		return ((( standard - self.limit["min"] ) / (self.limit["max"] - self.limit["min"])) * (self.outputMax - self.outputMin) ) + self.outputMin		

cols = ["Lagged Skelton", "Crakehill", "Skip Bridge", "Westwick", "Snaizeholmethe", "Arkengarthdale", "East Cowton", "Malham Tarn", "Skelton"]
colsAvg = [c+" Avg" for c in cols[1:]]

d = DataSet(
        filename = "Dataset", 
        limit = {"min": 0, "max": 1},
        lagby= {"Crakehill":1, "Skip Bridge":1, "Westwick":1, "Snaizeholmethe":1, "Arkengarthdale":1, "East Cowton":1, "Lagged Skelton":1, "Malham Tarn":1}, #'dict(zip(cols[:-1], [-1,-1,-1,-1,-1,-1,-1,-2])),
		moveAvg = 5,
		dateType = "j",
		stdOut = 3
    )

trainingData, validationData, testingData = d.getDataset(colsAvg+cols)

print(trainingData)

m = Model(
        network = {
                "inputs": 16,
                "hiddenLayers": [20],
                "outputs": 1
        },
        activation = "sigmoid",
		augments = {"momentum": True, "weightDecay": True, "annealing": True}
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

trainingInfo, validationInfo, epocs = m.train(
    trainingParams={
        "epocs": 500,
        "batch_size": 2,
        "learning_rate": [0.5,0.01],
        "validation_error_check_freq": 50,
        "validation_error_increase_limit": 1.03
    },
    trainingData= trainingData,
    validationData=validationData,
    inverseStandardise = d.inverseStandardise
)

print("TRAINING ERROR")
atr = m.test(dataset = trainingData, inverseStandardise = d.inverseStandardise)
print( atr)

print("TESTING ERROR")
ate = m.test(dataset = testingData, inverseStandardise = d.inverseStandardise)
print( ate) 

print("TOTAL TIME TAKEN:", time.time() - startTime, "SECS, FOR", epocs, "EPOCS")

# used to calculate final actual outputs
finalPlot = np.vstack([trainingData, validationData, testingData])
finalActual = np.empty((len(finalPlot),))
for i, row in enumerate (finalPlot):
	correctOutput = row[-1]
	inputs = row[:-1]
	actualOutput = m.forwardProp(inputs)[-1][1]
	finalActual[i] = actualOutput

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle("Error for training and validation sets")

# plot expected vs actual scatter plot
ax1.plot(finalActual, label="Observed" )
ax1.plot(finalPlot[:,-1], label="Wanted")

# plot training and validation graphs
ax2.plot(trainingInfo[trainingInfo != 0], label="Training")
ax2.plot(validationInfo[validationInfo != 0], label="Validation")

ax2.set(xlabel='epocs', ylabel='error')


ax2.legend(loc='best')

plt.show()