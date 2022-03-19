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

        layers = np.empty( (lenAllInputs-1,),  dtype=object )

        for i in range(1, lenAllInputs):
            numWeights = (numAllInputs[i-1]+1) * numAllInputs[i] 
            layer = np.random.uniform(low=-2/numAllInputs[i-1], high=2/numAllInputs[i-1], size=(numAllInputs[i], numAllInputs[i-1]+1))
            layers[i-1] = layer

        self.allLayerWeights = np.array([ np.array([[1,3,6], [-6,6,5]]), np.array([-3.92,2,4])])#layers

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

    def forwardPass(self, initInputs):
        layerInpOut = np.empty( (self.numLayers+1,), dtype=object )
        layerInpOut[0] = initInputs

        for i in range(1, self.numLayers+1):
            inputs = np.concatenate( [ [1], layerInpOut[i-1]] )
            output = np.dot(self.allLayerWeights[i-1], inputs)
            layerInpOut[i] = np.vectorize(self.activationFunctions)(output)

        return layerInpOut

    def getLayerWeights(self):
        return self.allLayerWeights


if __name__ == "__main__":
    m = Model(
            network = {
                    "inputs": 2,
                    "hiddenLayers": [2],
                    "outputs": 1
            },
            activation = "sigmoid"

        )

    for item in m.getLayerWeights():
        print(item)
        
    print( m.forwardPass(initInputs = np.array([1, 0])) )

