import numpy as np
import math

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
            layers[i-1] = layer

            # print("L",layer)
            nodeLayers[i-1] = np.delete(layer, [ i for i in range(layer.size) if i % 3 == 0])
            # print("N",nodeLayers[i-1])

        # self.allLayerWeights = np.array([ np.array([[1,3,6], [-6,6,5]]), np.array([-3.92,2,4])], dtype=np.ndarray)
        self.allLayerWeights = layers
        # for i in self.allLayerWeights:
        #     print(i.shape)
        # self.nodeWeights = np.array([ np.array([[3,6], [6,5]]), np.array([[2,4]])], dtype=np.ndarray)
        self.nodeWeights = nodeLayers
        # for i in self.nodeWeights:
        #     print(i.shape)
        # print("NWEIGHTS",nodeLayers)

    def activationFunctions(self, value):
        if self.activation == "sigmoid":
            return 1 / (1 + math.exp(-value))
        elif self.activation == "relu":
            return value if value > 0 else 0
        elif self.activation == "none":
            return value

    def activationDifferentials(self, value):
        if self.activation == "sigmoid":
            return value * (1 - value)
        elif self.activation == "relu":
            return 0 if value < 0 else 1 
        elif self.activation == "none":
            return value

    def forwardProp(self, initInputs):
        inpOutSize = self.numLayers+1

        layerInpOut = np.empty( (inpOutSize,), dtype=np.ndarray )
        layerInpOut[0] = initInputs

        for i in range(1, inpOutSize):
            inputs = np.concatenate( [ [1], layerInpOut[i-1]] )
            output = np.dot(self.allLayerWeights[i-1], inputs) 
            layerInpOut[i] = np.vectorize(self.activationFunctions)( np.array([output]) if np.isscalar(output) else output ) 

        return layerInpOut

    def calculateDelta(self, layerInpOut, correctOutput):

        inverseInputs = np.vectorize(self.activationDifferentials, otypes=[np.ndarray])( layerInpOut ) 
        outputDelta = (correctOutput - layerInpOut[-1]) * inverseInputs[-1]
        deltaValues = np.empty((self.numLayers,) ,dtype=np.ndarray)
        deltaValues[-1] = outputDelta

        for i in range(self.numLayers-1, -1, -1):

            # print("weights",self.nodeWeights[i])
            # print("deltas",deltaValues, i)

            nodeDelta = np.dot( self.nodeWeights[i], deltaValues[i][0] )[0]

            # print("dot",nodeDelta)

            nodeDelta = nodeDelta * inverseInputs[i]

            # print("delt",nodeDelta)
            # print("allvals", deltaValues )

            if i != 0:
                deltaValues[i-1] = nodeDelta

        return deltaValues

    def backwardsProp(self):
        ...

    def train(self):
        layerInpOut = self.forwardProp(np.array([1, 0]))
        deltaValues = self.calculateDelta(layerInpOut, 1)
        print(deltaValues)
        # print(layerInpOut)
        ...

    def getLayerWeights(self):
        return self.allLayerWeights


if __name__ == "__main__":
    m = Model(
            network = {
                    "inputs": 2,
                    "hiddenLayers": [2, 2],
                    "outputs": 1
            },
            activation = "sigmoid"

        )

    # for item in m.getLayerWeights():
    #     print(item)
        
    # print( m.forwardProp(initInputs = np.array([1, 0])) )

    m.train()
