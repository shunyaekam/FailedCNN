import random
import math
import csv


#Each individual node
class node:
  #The value fed into the node
  input = 0
  #The marking value of the node
  marker = (0, 0)

  #ReLu function for the output of the node
  def compute(self):
    if self.input < 0:
      return 0.01*self.input
    else:
      return self.input


#Softmax function
def softmax(inputs):
  temp = [math.exp(v) for v in inputs]
  total = sum(temp)
  return [t / total for t in temp]


#Initialization function
def initialize(input, numlayers, layer, output):
  global weights
  global nodes
  global gradients
  gradients = []
  weights = []
  nodes = []
  b = []
  #Initializing input nodes
  for i in range(input):
    b.append(node())
    b[i].marker = (0, i)
  nodes.append(b)
  b = []
  #Initializing middle layers
  for o in range(numlayers):
    for i in range(layer):
      b.append(node())
      b[i].marker = (1 + o, i)
    nodes.append(b)
    b = []
  #Initializing output layers
  for i in range(output):
    b.append(node())
    b[i].marker = (numlayers + 1, i)
  nodes.append(b)
  #Initializing weights between all the layers
  for i in range(1, len(nodes)):
    b = []
    for o in nodes[i]:
      l = []
      for p in nodes[i - 1]:
        l.append(random.random())
      b.append(l)
    weights.append(b)

def multiplyList(myList):
 
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result

#Running the network
def perform(values):
  global nodes
  global weights
  #Fixing the values to the input nodes
  for i in zip(nodes[0], values):
    i[0].input = i[1]
  #Computing values for each node
  for i in range(1, len(nodes)):
    for o in nodes[i]:
      oindex = nodes[i].index(o)
      k = []
      for l in nodes[i - 1]:
        lindex = nodes[i - 1].index(l)
        k.append(weights[i - 1][oindex][lindex] * l.compute())
      o.input = sum(k)
  #Computing the last nodes
  j = []
  for i in nodes[-1]:
    j.append(i.compute() * 0.000000001)
  #Applying softmax to the final nodes
  return j, softmax(j)

#Loss function
def loss(result, values):
  #Finding if we are just given the correct index value or a list of output values, and then converting to a list of preferred output values
  if len(values) == 1:
    c = values[0]
    values = [0 for i in range(0,len(result[1]))]
    values[c] = 1
  #Performing the loss function loss = (expectedvalue - returnedvalue)^2
  for i in range(0,len(nodes[-1])):
    values[i] = (values[i] - result[1][i])**2
  loss = sum(values)
  presoftmax = [0 for i in range(0,len(result[1]))]
  losspresoftmax = presoftmax.copy()
  presoftmax[c] = sum(result[0])
  for i in range(0,len(nodes[-1])):
    losspresoftmax[i] = (presoftmax[i] - result[0][i])**2
  if max(result[1]) == values.index(values[c]):
    b = 1
  else:
    b = 0
  return loss, values, presoftmax, losspresoftmax, result[0], b

#Calculating gradients for each weight
def gradient(loss, weights):
  #Derivative of ReLu function
  def ReLuderivative(value):
    if value <= 0:
      return 0.1
    return 100
  #Derivative of Cost function
  def CostDerivative(IntendedValues, ResultValues):
    Values = []
    for i in range(0, len(weights[-1])):
      Values.append(-2*(IntendedValues[i]-ResultValues[i]))
    return sum(Values)
    
  for i in range(len(weights)-1, 0, -1):
    #Layers
    k = []
    if i == len(weights):
      #Assigning gradients to weights connected directly to the output layer
      for u in range(0, len(weights[-1])):
        #Nodes
        j = []
        for p in range(0,len(weights[-1][u])):
          #Weights
          j.append(ReLuderivative((-2*(loss[2][u]-loss[4][u]))*nodes[i-1][p].compute())*(-2*(loss[2][u]-loss[4][u]))*nodes[i-1][p].compute() + 1)
        k.append(j)
    else: 
      for u in range(0, len(weights[i])):
        #Nodes
        e = CostDerivative(loss[2], loss[4])
        j = []
        #All the functions to be multiplied
        for p in range(0, len(weights[i][u])):
          t = []
          #Weights 
          for o in range(0, len(weights) - i):
            #Getting all values of weights from previous layers (derivatives)
            if o == 0:
              #Appending the values of only the weights that connect to the preceding node
              l = []
              for w in weights[i]: # i+1 --> i
                l.append(w[u])
              t.append(sum(l))
            else:
              t.append(sum([sum(x) for x in weights[i+1]]))
          t.append(nodes[i-1][p].compute())
          t.append(e)
          #List multiplication for the gradient of the weight
          t = multiplyList(t)
          j.append(ReLuderivative(t)*t)
          j = softmax(j)
        k.append(j)
    gradients.append(k)

def learn(gradients, weights, learningrate):
  for i in range(0,len(gradients)):
    for u in range(0, len(gradients[i])):
      for p in range(0, len(gradients[i][u])):
        weights[i][u][p] -= learningrate*gradients[i][u][p]
        
#Setup

initialize(784, 2, 16, 10)
filename = 'mnist_test.csv'
for x in range(100):
  gradients = []
  with open(filename, 'r') as csvfile:
    datareader = list(csv.reader(csvfile))
    i = datareader[random.randint(0, 234)]
    for o in i:
      i[i.index(o)] = int(o)
    b = perform(i[1:])
    gradient(loss(b, [i[0]]), weights)
    learn(gradients, weights, 0.1)
  a = loss(b, [i[0]])
  print(a[0])
  print(a[1])
  print(a[2])
  print(a[3])
  print(a[4])
  print(a[5])
  
  
