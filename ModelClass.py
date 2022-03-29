import numpy as np
import math
import time 
import copy

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
            self.activationDifferentials = lambda value: 0 if value < 0 else 1 
        elif self.activation == "none":
            self.activationFunctions = lambda value: value 
            self.activationDifferentials = lambda value: value

    # def activationFunctions(self, value):
    #     if self.activation == "sigmoid":
    #         return 1 / (1 + math.exp(-value))
    #     elif self.activation == "relu":
    #         return value if value > 0 else 0
    #     elif self.activation == "none":
    #         return value

    # def activationDifferentials(self, value):
    #     if self.activation == "sigmoid":
    #         return value * (1 - value)
    #     elif self.activation == "relu":
    #         return 0 if value < 0 else 1 
    #     elif self.activation == "none":
    #         return value

    def forwardProp(self, initInputs):
        inpOutSize = self.numLayers+1
        layerInpOut = np.empty( (inpOutSize,), dtype=np.ndarray )
        layerInpOut[0] = np.concatenate( [ [1], initInputs] ) #initInputs

        for i in range(1, inpOutSize):
            # inputs = np.concatenate( [ [1], layerInpOut[i-1]] )
            # print("weights",self.allLayerWeights[i-1])
            # print("inp",layerInpOut[i-1])
            layerOutSize = len(self.allLayerWeights[i-1])+1

            output = np.dot(self.allLayerWeights[i-1], layerInpOut[i-1]) #np.dot(self.allLayerWeights[i-1], inputs) 
            # print("OUT",output)
            outputActive = np.empty((layerOutSize,)) #np.empty((len(self.allLayerWeights[i-1]),)) 
            outputActive[0] =1
            for j in range (1, layerOutSize):
                # print(output[j-1])
                outputActive[j] = self.activationFunctions(output[j-1])

            layerInpOut[i] = outputActive

        return layerInpOut

    def calculateDelta(self, layerInpOut, correctOutput, deltaInpStore):

        initialDelta = ((correctOutput - layerInpOut[-1]) * self.activationDifferentials(layerInpOut[-1][1]))[1] #inverseInputs[-1]
        # print(layerInpOut)
        # print(layerInpOut[-2])
        # print(initialDelta)

        # print(deltaInpStore)

        deltaInpStore[-1] += initialDelta*layerInpOut[-2]
        # print( deltaInpStore )

        # print(  )

        # deltaValues = np.empty((self.numLayers,) ,dtype=np.ndarray)
        # deltaValues[-1] = outputDelta[1:]
        # deltaInpStore = np.copy(self.allLayerWeights) * 0

        # print("ALL", layerInpOut)

        for i in range(self.numLayers-1, 0, -1):
            nodeDelta = np.dot( self.nodeWeights[i], initialDelta )[0]
            # print(nodeDelta, "\n",[self.activationDifferentials(node) for node in layerInpOut[i]] )

            nodeDelta *= np.array([self.activationDifferentials(node) for node in layerInpOut[i][1:]] ) #inverseInputs[i]


            # print("INPUTS", layerInpOut[ self.numLayers-i-1 ])
            # print("allD", initialDelta)
            # print("d", nodeDelta)
            # print("inpMULT", layerInpOut[ self.numLayers-i-1 ] * nodeDelta)

            for j, delt in enumerate(nodeDelta):
                # print("delt", delt)
                # print("inpt", layerInpOut[ self.numLayers-i-1 ])
                # print("store B", deltaInpStore)
                deltaInpStore[self.numLayers-i-1][j] += layerInpOut[ self.numLayers-i-1 ] * delt
                # print("store A", deltaInpStore)
                # print()


            # print("STORE", deltaInpStore)

            initialDelta = nodeDelta

   
            


        return deltaInpStore

    def backwardsProp(self, learningRate, deltaInpStore): #layerInpOut, deltaValues):
        newAllLayerWeights = self.getLayerWeights()
        oldWeights = self.getOldLayerWeights()
        self.setOldLayerWeights(newAllLayerWeights)
        # self.oldLayerWeights = copy.deepcopy(newAllLayerWeights)
        for i, layer in enumerate(self.allLayerWeights): 
            for j, node in enumerate(layer ): 
                # print("cur", node)
                # print("old",oldWeights[i][j])
                # print("delta", deltaValues[i][j] )
                # print("input", layerInpOut[i])

                # print("WEIGHTSICAREABOUT", deltaValues[i][j] * layerInpOut[i])

                newWeights = ((node - oldWeights[i][j])* 0.9) + node + ((learningRate * deltaInpStore[i][j])) #((learningRate * deltaValues[i][j]) * layerInpOut[i])
                newAllLayerWeights[i][j] = newWeights

        self.setAllLayerWeights(newAllLayerWeights)

        # self.allLayerWeights = newAllLayerWeights

        # print("AFTER",self.allLayerWeights[0][0])

    def train(self, trainingParams, trainingData, validationData, inverseStandardise):

        trainingError = np.zeros((trainingParams["epocs"],))
        validationError = np.zeros((trainingParams["epocs"],))
        validationTracker = 1

        # deltaStore = np.zeros_like(self.allLayerWeights)
        # inpOutStore = np.zeros( (self.numLayers+1,), dtype=np.ndarray )
        deltaInpStore = np.copy(self.allLayerWeights) * 0
        # print("STORE", deltaInpStore)
        batchCounter = 0

        for i in range(trainingParams["epocs"]):

            lr = trainingParams["learning_rate"][1] + ( (trainingParams["learning_rate"][0] - trainingParams["learning_rate"][1]) * \
                                                        (1 -( 1 /  (1 + math.exp(10 - ((20 * (i)) / trainingParams["epocs"]))))))
            # print(lr)
            # lr = trainingParams["learning_rate"][0]
            b= 0
            for row in trainingData:
                correctOutput = row[-1]
                inputs = row[:-1]
                layerInpOut = self.forwardProp(inputs)
                deltaInpStore = self.calculateDelta(layerInpOut, correctOutput, deltaInpStore)
                # print(layerInpOut)

                # for j, layer in enumerate(deltaValues):
                #     for k, delt in enumerate(layer):
                        # print("delt", delt)
                        # print("input", layerInpOut[j])
                        # print("curitem", deltaInpStore[j][k])

                # deltaInpStore += deltaValues

                # print("running avg", deltaInpStore)

                # deltaStore += deltaValues
                # inpOutStore += layerInpOut
                batchCounter+=1
                # print(deltaValues)

                if (batchCounter % trainingParams["batch_size"] == 0):
                    batchCounter = 0
                    self.backwardsProp(lr, deltaInpStore/trainingParams["batch_size"])#inpOutStore/trainingParams["batch_size"], deltaStore/trainingParams["batch_size"])
                    # deltaStore *= 0
                    # inpOutStore *= 0
                    deltaInpStore *= 0

        
                # b += 1
                # if b ==5:
                #     return
            if (i % trainingParams["validation_error_check_freq"] == 0):
                newValidationErr = self.test(validationData)
                # to test if validation error has increaced over the limit
                if (newValidationErr < validationTracker * trainingParams["validation_error_increase_limit"]):
                    validationTracker = newValidationErr
                else:
                    break

            trainingError[i] = self.test(trainingData, inverseStandardise)
            validationError[i] =self.test(validationData, inverseStandardise)

        return trainingError, validationError, i+1

    def test(self, dataset, inverseStandardise = lambda x: x):

        total = 0
        for row in dataset:
            correctOutput = row[-1]
            inputs = row[:-1]
            actualOutput = self.forwardProp(inputs)[-1][1]
            # print(self.forwardProp(inputs)[-1])
            # print("ACTUAL",actualOutput)
            # if (total > 5):
            #     i = 5/0
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

    # for item in m.getLayerWeights():
    #     print(item)
        
    # print( m.forwardProp(initInputs = np.array([1, 0])) )
    # print("BEFORE TRAINING")

    # print(m.test(
    #     dataset= np.array([[1, 0, 1]])
    # ))

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