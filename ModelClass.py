import numpy as np
import math
import copy

class Model:
    def __init__(self, network, activation, augments):
        self.numInputs = network["inputs"]
        self.numLayers = len( network["hiddenLayers"] ) + 1
        self.activation = activation

        self.momentum = augments["momentum"]
        self.weightDecay = augments["weightDecay"]
        self.annealing = augments["annealing"]

        numAllInputs = [network["inputs"]] + network["hiddenLayers"] + [network["outputs"]]
        lenAllInputs = len(numAllInputs)

        # used to initialize random weights 
        layers = np.empty( (lenAllInputs-1,),  dtype=np.ndarray  )
        nodeLayers = np.empty( (lenAllInputs-1,),  dtype=np.ndarray )
        for i in range(1, lenAllInputs):
            layer = np.random.uniform(  low=-2/numAllInputs[i-1], high=2/numAllInputs[i-1], \
                                        size=(numAllInputs[i], numAllInputs[i-1]+1))
            layers[i-1] = layer
            nodeLayers[i-1] = np.delete(layer, [ j for j in range(layer.size) if j % (numAllInputs[i-1]+1) == 0])
        self.allLayerWeights = layers
        self.nodeWeights = nodeLayers

        # store old weights for momentum improvement
        self.oldLayerWeights = copy.deepcopy(self.allLayerWeights)

        # layer activation functions
        if self.activation == "sigmoid":
            self.activationFunctions = lambda value: 1 / (1 + math.exp(-value))
            self.activationDifferentials = lambda value: value * (1 - value)
        elif self.activation == "relu":
            self.activationFunctions = lambda value: np.maximum(value, 0)
            self.activationDifferentials = lambda value: np.greater(value, 0.).astype(np.float32)
        elif self.activation == "tanh":
            self.activationFunctions = lambda value: (math.exp(value) - math.exp(-value)) / \
                                                        (math.exp(value) + math.exp(-value))
            self.activationDifferentials = lambda value: 1 - ((math.exp(value) - math.exp(-value)) / \
                                                        (math.exp(value) + math.exp(-value)))**2
        elif self.activation == "none":
            self.activationFunctions = lambda value: value 
            self.activationDifferentials = lambda value: value

    # apply dot product between weights and inputs
    # return a 2d array containing input layer and output of each layer
    def forwardProp(self, initInputs):
        inpOutSize = self.numLayers+1
        layerInpOut = np.empty( (inpOutSize,), dtype=np.ndarray )

        # concatenate the input for the bias
        layerInpOut[0] = np.concatenate( [ [1], initInputs] ) 

        for i in range(1, inpOutSize):
            layerOutSize = len(self.allLayerWeights[i-1])+1

            # dot product of weights and inputs
            output = np.dot(self.allLayerWeights[i-1], layerInpOut[i-1]) 

            # activate each output
            outputActive = np.empty((layerOutSize,), dtype=float) 
            outputActive[0] =1
            for j in range (1, layerOutSize):
                outputActive[j] = self.activationFunctions(output[j-1])

            # save layer outputs
            layerInpOut[i] = outputActive

        return layerInpOut

    # calculate delta values then multiply delta values with inputs
    # return a 3d array containing the running sum of (delta values * input values)
    def calculateDelta(self, layerInpOut, correctOutput, deltaInpStore, weightDecayAvg, weightDecayParam):

        # delta for output node with WEIGHT DECAY
        initialDelta = (correctOutput - layerInpOut[-1]) + \
                        ((weightDecayAvg * weightDecayParam) if self.weightDecay else 0)

        initialDelta = (initialDelta * self.activationDifferentials(layerInpOut[-1][1]))[1] 

        # delta * input for output node
        deltaInpStore[-1] += initialDelta*layerInpOut[-2]

        for i in range(self.numLayers-1, 0, -1):
            # calculate delta for each node
            nodeDelta = np.dot( self.nodeWeights[i], initialDelta )[0]
            nodeDelta *= np.array([self.activationDifferentials(node) for node in layerInpOut[i][1:]])

            # multiply with inputs for each delta 
            for j, delt in enumerate(nodeDelta):
                deltaInpStore[self.numLayers-i-1][j] += (layerInpOut[ self.numLayers-i-1 ] * delt)

            initialDelta = nodeDelta

        return deltaInpStore

    # update weights based on running sum of (delta * inputs)
    def backwardsProp(self, deltaInpStore):

        newAllLayerWeights = self.getLayerWeights()

        # update old weights to current weights
        oldWeights = self.getOldLayerWeights()
        self.setOldLayerWeights(newAllLayerWeights)

        for i, layer in enumerate(self.allLayerWeights): 
            for j, node in enumerate(layer ): 

                # Update each weight 
                newWeights = node + deltaInpStore[i][j]

                # apply MOMENTUM
                if self.momentum: 
                    newWeights += ((node - oldWeights[i][j])* 0.9)

                newAllLayerWeights[i][j] = newWeights

        self.setAllLayerWeights(newAllLayerWeights)

    # calculate average squared weight
    # return the average weight squared
    def getWeightDecayAvg(self):
        allw = self.getLayerWeights()
        weightDecayAvg = 0
        weightDecayAvgSize = 0
        for layer in allw:
            weightDecayAvg += np.sum(layer**2)
            weightDecayAvgSize += layer.size
        weightDecayAvg /= 2 * weightDecayAvgSize
        return weightDecayAvg

    # cycle through epocs and training data to train model
    # return the number of epocs taken along with error for training and validation
    def train(self, trainingParams, trainingData, validationData, inverseStandardise):

        # used to plot validation and training error
        trainingError = np.zeros((trainingParams["epocs"],))
        validationError = np.zeros((trainingParams["epocs"],))
        validationTracker = 1

        # used to store the sum of (input * delta) values for BATCH LEARNING
        deltaInpStore = np.copy(self.allLayerWeights) * 0
        batchCounter = 0

        # initialise WEIGHT DECAY average
        weightDecayAvg = self.getWeightDecayAvg()

        lr = trainingParams["learning_rate"][0]

        for i in range(trainingParams["epocs"]):

            # update WEIGHT DECAY parameter
            weightDecayParam = 1 / (lr * (i+1))

            for row in trainingData:
                correctOutput = row[-1]
                inputs = row[:-1]
                layerInpOut = self.forwardProp(inputs)

                # update num of (input * delta)
                deltaInpStore = lr * self.calculateDelta(layerInpOut, correctOutput, deltaInpStore, weightDecayAvg, weightDecayParam)
                batchCounter+=1

                # BATCH LEARNING implementation  
                if (batchCounter % trainingParams["batch_size"] == 0):
                    # reset batch counter and apply back prop
                    batchCounter = 0
                    self.backwardsProp(deltaInpStore/trainingParams["batch_size"])

                    # reset (input * delta) sum
                    deltaInpStore *= 0

                    # recalculate average squared weight 
                    weightDecayAvg = self.getWeightDecayAvg()

            if (i % trainingParams["validation_error_check_freq"] == 0):
                newValidationErr = self.test(validationData)

                # to test if validation error has increaced over the limit
                if (newValidationErr < validationTracker * trainingParams["validation_error_increase_limit"]):
                    validationTracker = newValidationErr
                else:
                    break

            # record training and validation error
            trainingError[i] = self.test(trainingData, inverseStandardise)
            validationError[i] = self.test(validationData, inverseStandardise)

            # ANNEALING IMPLEMENTATION 
            if self.annealing:
                lr = trainingParams["learning_rate"][1] + ( (trainingParams["learning_rate"][0] - trainingParams["learning_rate"][1]) * \
                                                            (1 -( 1 /  (1 + math.exp(10 - ((20 * (i)) / trainingParams["epocs"]))))))

        return trainingError, validationError, i+1

    # calculate the error of a dataset
    # return root mean squared error
    def test(self, dataset, inverseStandardise = lambda x: x):

        total = 0
        for row in dataset:
            correctOutput = row[-1]
            inputs = row[:-1]
            actualOutput = self.forwardProp(inputs)[-1][1]
            total += (inverseStandardise(actualOutput) - inverseStandardise(correctOutput))**2
        
        return (total/len(dataset))**0.5

    def getLayerWeights(self):
        return self.allLayerWeights

    def getOldLayerWeights(self):
        return self.oldLayerWeights

    def setAllLayerWeights(self, val):
        self.allLayerWeights = val

    def setOldLayerWeights(self, val):
        self.oldLayerWeights = copy.deepcopy(val)

# for testing
if __name__ == "__main__":
    np.set_printoptions(precision=10)

    m = Model(
            network = {
                    "inputs": 2,
                    "hiddenLayers": [2],
                    "outputs": 1
            },
            activation = "sigmoid",
            augments = {"momentum": False, "weightDecay": False, "annealing": False}
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