class NeuralNetwork:
  #Initializing the Neural Network
  def __init__(self, inputlayers, hiddenlayers, hiddenlayersize, outputlayers):
    self.inputs = inputlayers
    self.hiddenlayers = hiddenlayers
    self.hiddenlayersize = hiddenlayersize
    self.outputs = outputlayers
    #Initializing the input and output nodes
    nodes = [[0 for y in range(inputlayers)], [0 for y in range(outputlayers)]]
    #Initializing the hidden layers
    for x in range(hiddenlayers):
      nodes.insert(1, [0 for y in range(hiddenlayersize)])
    self.nodes = nodes
    #Initializing the output weights
    weights = [[0 for y in range(outputlayers)]]
    #Initializing the hidden layers
    for x in range(hiddenlayers):
      weights.insert(0, [0 for y in range(hiddenlayersize)])
    #Initializing each x,y,z value of weights to be a random number in between 0,1
    import random
    for x in range(0,len(weights)):
      for y in range(0,len(weights[x])):
        weights[x][y] = [random.uniform(0.2,1) for _ in range(len(nodes[x]))]
    self.weights = weights

  #ReLU function
  def ReLU(self, n):
    if n < 0:
      return 0.01 * n
    return n

  #Softmax function
  def softmax(self, nlist):
    import math
    #Values become too large, thus need to be divided
    temp = [math.exp(v/10000) for v in nlist]
    total = sum(temp)
    return [t / total for t in temp]

  #Forward propogation in the neural network
  def FeedForward(self, inputs):
    '''nodes = self.nodes
    weights = self.weights'''
    #Setting the input nodes (f(x)\/n = i\/n)
    self.nodes[0] = inputs
    #Propogating forward through the network using the input nodes
    for x in range(1, len(self.nodes)):
      for y in range(0, len(self.nodes[x])):
        #Calculating the values for each node (f(x)\/n = w\/(x-1, y)*f(x-1(y-1)) +... w\/(x-1, y-n)*f(x-1(y-1)))
        PreActivationValue = 0
        for z in self.weights[x-1][y]:
          PreActivationValue += (z * self.nodes[x-1][self.weights[x-1][y].index(z)])
        self.nodes[x][y] = self.ReLU(PreActivationValue)
    #Applying Softmax to output nodes
    self.ProbabilisticApproximation = self.softmax(self.nodes[-1])
    return self.nodes, self.ProbabilisticApproximation
  def Learn(self, expected, learningrate):
    nodes = self.nodes
    weights = self.weights
    #Calculate Loss
    def Loss(expected):
      expectedValues = [0 for y in nodes[-1]]
      expectedValues[expected] = 1
      return [a_i - b_i for a_i, b_i in zip(expectedValues, self.ProbabilisticApproximation)], expectedValues
    #Derivative for ReLU
    def ReLUderivative(value):
      if value <= 0:
        return 0.1
      return 1
    #Derivative for Softmax
    def Softmaxderivative(value):
      return value*(1-value)
    #Update function
    def update(weights, gradients, learningrate):
      for x in weights:
        for y in weights:
          weights[x][y] = [a_i - learningrate*b_i for a_i, b_i in zip(weights[x][y], gradients[x][y])]
    #Calculate gradients
    def Gradients():
      #Values for the Gradients
      Gradientvalues = weights.copy()
      #Running the Loss function
      Expectedvalues = Loss(expected)[1]
      #Running through all the weights (x,y,z) and calculating the Gradients
      for x in range(0, len(weights)):
        for y in range(0, len(weights[x])):
          for z in range(0, len(weights[x][y])):
            Gradientzvalue = 1
            #Multiplying dprev(x)/dweight(x,y,z)
            Gradientzvalue *= nodes[x][z]
            #Going through all the other layers
            for i in range (0, len(weights) - y):
              #Iterating through the last layer (Cost function, Sigmoid)
              if i == 0:
                #If this is the last layer
                if y == weights.index(weights[-1]):
                  #Multiplying the Cost and Softmax Derivatives for only the one weight
                  Gradientzvalue *= Softmaxderivative(self.ProbabilisticApproximation[z]) # z --> z-1
                  Gradientzvalue*= -2*(Expectedvalues[z] - self.ProbabilisticApproximation[z]) # z --> z-1
                  break
                #If this isn't the last layer (Multiplying the Cost and Softmax Derivatives for all the weights)
                Gradientzvalue *= sum([Softmaxderivative(Resultvalue) for Resultvalue in self.ProbabilisticApproximation])
                Gradientzvalue *= sum([-2*(yhat - Resultvalue) for yhat, Resultvalue in zip(Expectedvalues, self.ProbabilisticApproximation)])
              #If we are one layer away from the weight (Multiplying only the weights that connect to the connected node)
              elif weights.index(weights[-1])-y-i == 1: # ==1 --> ==2
                Gradientzvalue*= sum(weights[x+1][z])
                break
              #If we are somewhere in between
              else:
                Gradientzvalue*= sum([sum(y) for y in weights[weights.index(weights[-1])-i]])
          Gradientvalues[x][y][z] = Gradientzvalue
      return Gradientvalues
    print("Loss:", Loss(expected)[0])
    gradients = Gradients()
    update(weights, gradients, 0.1)
                
#Initializing the test environment
#NeuralNetwork the size of the images
NeuralNetwork = NeuralNetwork(784, 2, 16, 10)
import csv, random
#Loading the file
filename = 'mnist_test.csv'
for e in range(0,5):
  print(e)
  with open(filename, 'r') as csvfile:
      datareader = list(csv.reader(csvfile))
      #Selecting a random example
      data = datareader[random.randint(0, 234)]
      #Converting the strings into numbers
      for num in data[1:]:
        data[data.index(num)] = float(num)/16
      #Executing the function
      #print(NeuralNetwork.weights[0])
      NeuralNetwork.FeedForward(data[1:])
      NeuralNetwork.Learn(int(data[0]), 0.1)
