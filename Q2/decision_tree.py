from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        self.tree = []
        #self.tree = {}
        pass

    def getMean(self, X, col):
        total = 0
        for i in range(len(X)):
            total += X[i][col]
        return total/(len(X)*1.0)

    def getUnique(self, X, col):
        m = {}
        for i in range(len(X)):
            try:
                m[X[i][col]] += 1
            except:
                m[X[i][col]] = 1
        return list(m.keys())

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        rows = len(X)
        noFeatures = len(X[0])
        leaf = [[-1, y[np.random.randint(len(y))] , None, None]]
        if(rows <= 5 or len(set(y)) == 1):
          return leaf
       
        maxIG = -100000000
        splitIndex = -1
        splitValue = -1
        x_left = []
        y_left = []
        x_right = []
        y_right = []

        
        for j in range(len(X[0])):
            currInformationGain = -10000000000
            if isinstance(X[0][j], float) or isinstance(X[0][j], int) :
                # uniqueNums = self.getUnique(X,j)
                # splitVal = self.getMean(X,j)
                for i in range(0,1):
                    splitVal = X[np.random.randint(len(X))][j]
                    xLeft, xRight, yLeft, yRight = partition_classes(X, y, j, splitVal)
                    currInformationGain = information_gain(y, [yLeft, yRight])
                    if currInformationGain > maxIG:
                        maxIG = currInformationGain
                        splitIndex = j
                        splitValue = splitVal
                        x_left = xLeft
                        y_left = yLeft
                        x_right = xRight
                        y_right = yRight
            else:
                uniqueStrs = self.getUnique(X, j)
                # print(uniqueStrs)
                for string in uniqueStrs:
                    splitVal = string
                    xLeft, xRight, yLeft, yRight = partition_classes(X, y, j, splitVal)
                    InformationGain = information_gain(y, [yLeft, yRight])
                    if currInformationGain > maxIG:
                        maxIG = currInformationGain
                        splitIndex = j
                        splitValue = splitVal
                        x_left = xLeft
                        y_left = yLeft
                        x_right = xRight
                        y_right = yRight
            

        if(len(x_left) == 0 or len(x_right) == 0):
            return leaf

        leftSubTree = self.learn(x_left, y_left)
        rightSubTree = self.learn(x_right, y_right)
        if(len(leftSubTree) == 1):
          root = [[splitIndex, splitValue, 1, 2]]
        else:
          root = [[splitIndex, splitValue, 1, len(leftSubTree) + 1]]

        return root + leftSubTree + rightSubTree


    def find(self, xTest, i):
      nodeVal = self.tree[i]
      splitIndex = nodeVal[0]
      try:
        predictVal = float(nodeVal[1])
        if(splitIndex == -1):
          return predictVal
        leftIndex = int(nodeVal[2])
        rightIndex = int(nodeVal[3])
        if(xTest[splitIndex] <= predictVal):
          return self.find(xTest, i + leftIndex)
        if(xTest[splitIndex] > predictVal):
          return self.find(xTest, i + rightIndex)
      except:
        predictVal = nodeVal[1]
        if(splitIndex == -1):
          return predictVal
        leftIndex = int(nodeVal[2])
        rightIndex = int(nodeVal[3])
        if(xTest[splitIndex] == predictVal):
          return self.find(xTest, i + leftIndex)
        if(xTest[splitIndex] != predictVal):
          return self.find(xTest, i + rightIndex)

    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        classLabel = self.find(record, 0)
        return classLabel
