import numpy as np
import math
import time 
import copy
import matplotlib.pyplot as plt

class Model:
    def __init__(self, network, activation):
        
        self.numInputs = network["inputs"]
        self.numLayers = len( network["hiddenLayers"] ) + 1
        self.numOutputs = network["outputs"]
        self.activation = activation

        numAllInputs = [network["inputs"]] + network["hiddenLayers"] + [network["outputs"]]
        lenAllInputs = len(numAllInputs)

        layers = np.empty( (lenAllInputs-1,),  dtype=np.ndarray  )
        nodeLayers = np.empty( (lenAllInputs-1,),  dtype=np.ndarray )
        for i in range(1, lenAllInputs):
            layer = np.random.uniform(low=-2/numAllInputs[i-1], high=2/numAllInputs[i-1], size=(numAllInputs[i], numAllInputs[i-1]+1))
            # layer = np.ones((numAllInputs[i], numAllInputs[i-1]+1))
            layers[i-1] = layer

            nodeLayers[i-1] = np.delete(layer, [ j for j in range(layer.size) if j % (numAllInputs[i-1]+1) == 0])

        # self.allLayerWeights = np.array([ np.array([[1,3,4], [-6,6,5]]), np.array([[-3.92,2,4]])], dtype=np.ndarray)
        self.allLayerWeights = layers
        self.oldLayerWeights = copy.deepcopy(self.allLayerWeights)

        # self.nodeWeights = np.array([ np.array([[3,4], [6,5]]), np.array([[2,4]])], dtype=np.ndarray)
        self.nodeWeights = nodeLayers

        if self.activation == "sigmoid":
            self.activationFunctions = lambda value: 1 / (1 + math.exp(-value))
            self.activationDifferentials = lambda value: value * (1 - value)
        elif self.activation == "relu":
            self.activationFunctions = lambda value: value if value > 0 else 0
            self.activationDifferentials = lambda value: 0 if value < 0 else value
        elif self.activation == "none":
            self.activationFunctions = lambda value: value 
            self.activationDifferentials = lambda value: value

    def forwardProp(self, initInputs):
        inpOutSize = self.numLayers+1
        layerInpOut = np.empty( (inpOutSize,), dtype=np.ndarray )
        layerInpOut[0] = np.concatenate( [ [1], initInputs] ) 

        for i in range(1, inpOutSize):

            layerOutSize = len(self.allLayerWeights[i-1])+1

            output = np.dot(self.allLayerWeights[i-1], layerInpOut[i-1]) 

            outputActive = np.empty((layerOutSize,), dtype=float) 
            outputActive[0] =1
            for j in range (1, layerOutSize):
                outputActive[j] = self.activationFunctions(output[j-1])

            layerInpOut[i] = outputActive

        return layerInpOut

    def calculateDelta(self, layerInpOut, correctOutput, deltaInpStore, weightDecayAvg, weightDecayParam):

        ## WEIGHT DECAY IMPLEMENTATION 
        initialDelta = (((correctOutput - layerInpOut[-1]) + (weightDecayAvg * weightDecayParam) ) * self.activationDifferentials(layerInpOut[-1][1]))[1] 
        # print(deltaInpStore)
        # print("INITIAL", initialDelta)
        deltaInpStore[-1] += initialDelta*layerInpOut[-2]

        for i in range(self.numLayers-1, 0, -1):

            nodeDelta = np.dot( self.nodeWeights[i], initialDelta )[0]

            nodeDelta *= np.array([self.activationDifferentials(node) for node in layerInpOut[i][1:]])

            for j, delt in enumerate(nodeDelta):
                deltaInpStore[self.numLayers-i-1][j] += (layerInpOut[ self.numLayers-i-1 ] * delt)

            initialDelta = nodeDelta

        return deltaInpStore

    def backwardsProp(self, learningRate, deltaInpStore):

        newAllLayerWeights = self.getLayerWeights()

        oldWeights = self.getOldLayerWeights()
        self.setOldLayerWeights(newAllLayerWeights)

        for i, layer in enumerate(self.allLayerWeights): 
            for j, node in enumerate(layer ): 

                ## MOMENTUM IMPLEMENTATION
                newWeights = node + deltaInpStore[i][j] + ((node - oldWeights[i][j])* 0.9)
                newAllLayerWeights[i][j] = newWeights

        self.setAllLayerWeights(newAllLayerWeights)

    def getWeightDecayAvg(self):
        allw = self.getLayerWeights()
        weightDecayAvg = 0
        weightDecayAvgSize = 0
        for layer in allw:
            weightDecayAvg += np.sum(layer**2)
            weightDecayAvgSize += layer.size
        weightDecayAvg /= 2 * weightDecayAvgSize
        return weightDecayAvg

    def train(self, trainingParams, trainingData, validationData, inverseStandardise):

        trainingError = np.zeros((trainingParams["epocs"],))
        validationError = np.zeros((trainingParams["epocs"],))
        validationTracker = 1

        deltaInpStore = np.copy(self.allLayerWeights) * 0
        batchCounter = 0

        weightDecayAvg = self.getWeightDecayAvg()
        lr = trainingParams["learning_rate"][0]

        # tasda = []
        # weights = []
        # params = []

        for i in range(trainingParams["epocs"]):

            weightDecayParam = 1 / (lr * (i+1))
            # tasda.append(  weightDecayAvg* weightDecayParam )
            # weights.append(  weightDecayAvg )
            # params.append(  weightDecayParam )
            for row in trainingData:
                correctOutput = row[-1]
                inputs = row[:-1]
                layerInpOut = self.forwardProp(inputs)
                # print(weightDecayAvg)
                # if i+1 < 3:
                #     weightDecayAvg = 0
                deltaInpStore = lr * self.calculateDelta(layerInpOut, correctOutput, deltaInpStore, weightDecayAvg, weightDecayParam)

                batchCounter+=1

                ## BATCH LEARNING DECAY IMPLEMENTATION 
                if (batchCounter % trainingParams["batch_size"] == 0):
                    batchCounter = 0
                    self.backwardsProp(lr, deltaInpStore/trainingParams["batch_size"])
                    deltaInpStore *= 0
                    weightDecayAvg = self.getWeightDecayAvg()

            if (i % trainingParams["validation_error_check_freq"] == 0):
                newValidationErr = self.test(validationData)

                # to test if validation error has increaced over the limit
                if (newValidationErr < validationTracker * trainingParams["validation_error_increase_limit"]):
                    validationTracker = newValidationErr
                else:
                    break

            trainingError[i] = self.test(trainingData, inverseStandardise)
            validationError[i] = self.test(validationData, inverseStandardise)

            ## ANNEALING DECAY IMPLEMENTATION 
            lr = trainingParams["learning_rate"][1] + ( (trainingParams["learning_rate"][0] - trainingParams["learning_rate"][1]) * \
                                                        (1 -( 1 /  (1 + math.exp(10 - ((20 * (i)) / trainingParams["epocs"]))))))

        # plt.plot(tasda, label="test")
        # plt.plot(weights, label="avgs")
        # plt.plot(params, label="params")
        # plt.legend(loc='best')

        # plt.show()
        # return
        return trainingError, validationError, i+1

    def test(self, dataset, inverseStandardise = lambda x: x):

        total = 0
        for row in dataset:
            correctOutput = row[-1]
            inputs = row[:-1]
            actualOutput = self.forwardProp(inputs)[-1][1]
            total += (inverseStandardise(actualOutput - correctOutput))**2
        
        return (total/len(dataset))**0.5

    def getLayerWeights(self):
        return self.allLayerWeights

    def getOldLayerWeights(self):
        return self.oldLayerWeights

    def setAllLayerWeights(self, val):
        self.allLayerWeights = val

    def setOldLayerWeights(self, val):
        self.oldLayerWeights = copy.deepcopy(val)

if __name__ == "__main__":
    np.set_printoptions(precision=10)

    m = Model(
            network = {
                    "inputs": 2,
                    "hiddenLayers": [2],
                    "outputs": 1
            },
            activation = "sigmoid"

        )

    print(m.getLayerWeights())
    print()
    trainingInfo, validationInfo, epocs = m.train(
        trainingParams={
            "epocs": 100,
            "batch_size":1,
            "learning_rate": [0.1, 0.01],
            "validation_error_check_freq": 100,
            "validation_error_increase_limit": 1.05
        },
        trainingData= np.array([[1, 0, 1]]),
        validationData=np.array([[1, 0, 1]]),
        inverseStandardise = lambda x: x
    )

    print(m.getLayerWeights())

    print("\nAFTER TRAINING")
    test = m.test(dataset = np.array([[1, 0, 1]]))
    print( test )


    print("\nOUTPUT")
    out = m.forwardProp(initInputs = np.array([1, 0]))    
    print( out )


    test = m.calculateDelta(
        layerInpOut= np.array([np.array([1, 1, 0]), np.array([1, 0.982, 0.5]), np.array([1, 0.511])], dtype=np.ndarray),
        correctOutput= 1,
        deltaInpStore= np.array([np.array([[0, 0, 0],
                                            [ 0, 0 ,  0]] , dtype=float),
                                np.array([[ 0, 0,  0]], dtype=float)], dtype=np.ndarray),

        weightDecayAvg= 0,
        weightDecayParam= 0,
    )

    print(test)