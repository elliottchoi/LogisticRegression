import torch
import torch.functional as F
from torch.autograd import Variable

import numpy as np

#created multiple layers for a neural network using a logistic optimizer 

xy = np.loadtxt('/Users/elliottchoi/Desktop/Code_Repository/data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data=Variable(torch.from_numpy(xy[:,0:-1]))
y_data=Variable(torch.from_numpy(xy[:,[-1]]))

#prints the dimensions of the data
print(x_data.data.shape)
print(y_data.data.shape)

class Model(torch.nn.Module):

    #use of sigmoid
    def __init__(self):
        super(Model,self).__init__()
        #4 layers of back neural network
        self.layerOne=torch. nn.Linear(8,6)
        self.layerTwo =torch.nn.Linear(6, 4)
        self.layerThree = torch.nn.Linear(4, 2)
        self.layerFour = torch.nn.Linear(2, 1)

        #in the end of the data, we connecate 8 vars to 1 output

        #output is 0 or 1, so the appropriate use of a sigmoid
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self,x):
        out1=self.sigmoid(self.layerOne(x))
        out2 = self.sigmoid(self.layerTwo(out1))
        out3 = self.sigmoid(self.layerThree(out2))
        y_predict = self.sigmoid(self.layerFour(out3))

        return y_predict

#model object
model=Model()

#Use of Sigmoid means we want to use BCE loss
criterion=torch.nn.BCELoss(size_average=True)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range (1000):
    y_predict=model(x_data)

    loss=criterion(y_predict,y_data)
    print(epoch,loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


