import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# return pixel # in 256 array format given [x,y]
def get_xy(x,y):
	return 16*x+y

# sign function
def sign(num):
    if (num > 0):
        return 1
    else:
        return -1

# dot product two vectors
# w and x must be the same length
def dot(w,x):
    sum = 0
    if (len(w) != len(x)):
        return NaN
    else:
        for i in range(0,len(w)):
            sum = sum + x[i]*w[i]
        return sum

# scale val from [min,max] to [-1,1]
def scale(val,min,max):
	retval = (val-min)/(max-min)
	return 2*retval-1

# return value in [0,1] denoting measure of vertical symmetry
def get_symmetry(arr):
	total_symm = 0
	# calculate absolute difference between values of two reflected pixels
	# subtract from 2 and divide by 2 to get normalized value in [0,1] range
	for i in range(0,8):
		for j in range(0,16):
			total_symm += 1-abs(arr[get_xy(i,j)]-arr[get_xy(15-i,j)])
	total_symm /= 128
	return total_symm

# return avg intensity of pixels
def get_intensity(arr):
	total_intensity = 0
	for i in range(0,256):
		total_intensity += arr[i]
	total_intensity /= 256
	return total_intensity

# plot digits data on symmetry and intensity axes
def plot_points(data, y):
    colors = ['b','g','r','c','m','y','pink','orange','lime','deepskyblue']
    for i in range(0,len(data)):
        p_color = colors[int(y[i])]
        plt.plot(data[i][0],data[i][1],color=p_color,marker='.')
    plt.xlabel('Symmetry')
    plt.ylabel('Intensity')



##### main code #####

### obtaining data ###
random.seed(26)
df_train = pd.read_csv("../../../csci-4100/hw12/ZipDigitsTest.txt", sep = ' ',engine='python',header=None,usecols=range(0,257))
df_train = df_train.to_numpy()
df_test = pd.read_csv("../../../csci-4100/hw12/ZipDigitsTrain.txt", sep = ' ',engine='python',header=None,usecols=range(0,257))
df_test = df_test.to_numpy()

# find min and max symmetry and intensity values for scaling
df_scale = pd.read_csv("../../../csci-4100/hw12/ZipDigits.txt", sep = ' ',engine='python',header=None,usecols=range(0,257))
df_scale = df_scale.to_numpy()
feat = [] 	# stores symmetry and intensity values for each data point
feat1_data = []
feat2_data = []
for i in range(0,len(df_scale)):
	feat1_data.append(get_symmetry(df_scale[i][1:257]))
	feat2_data.append(get_intensity(df_scale[i][1:257]))
x0 = [1 for i in range(len(df_scale))] 	# append column of 1s to data
feat.append(x0)
feat.append(feat1_data)
feat.append(feat2_data)
X_big = np.array(feat)
min_symm = np.amin(X_big[1])
max_symm = np.amax(X_big[1])
min_inte = np.amin(X_big[2])
max_inte = np.amax(X_big[2])

# parse csv file - ../hw9/ZipDigitsTrain.txt
data_train = [] 	# stores symmetry and intensity values for each data point
for i in range(0,len(df_train)):
	# print(i)
	features = []
	features.append(scale(get_symmetry(df_train[i][1:257]),min_symm,max_symm))
	features.append(scale(get_intensity(df_train[i][1:257]),min_inte,max_inte))
	data_train.append(features)
data_train = np.array(data_train)
# parse csv file - ../hw9/ZipDigitsTest.txt
data_test = [] 	# stores symmetry and intensity values for each data point
for i in range(0,len(df_test)):
	features = []
	features.append(scale(get_symmetry(df_test[i][1:257]),min_symm,max_symm))
	features.append(scale(get_intensity(df_test[i][1:257]),min_inte,max_inte))
	data_test.append(features)
data_test = np.array(data_test)

y_train = df_train[:,0]
y_test = df_test[:,0]
y_train = [int(i) for i in y_train]
y_test = [int(i) for i in y_test]
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
classes = [0,1,2,3,4,5,6,7,8,9]


### set up dataloaders ###
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# custom Dataset and DataLoader classes
class MyDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i]
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data
        
train_data = MyDataset(data_train, y_train)
test_data = MyDataset(data_test, y_test)

trainloader = DataLoader(train_data, batch_size=128, shuffle=True)
testloader = DataLoader(test_data, batch_size=128, shuffle=True)

# NN setup
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # first fully connected layer
        self.fc1 = nn.Linear(2,128)
        
        # second fully connected layer that outputs 10 labels
        self.fc2 = nn.Linear(128,10)
        
    # forward propagation, x = data
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        # apply softmax to x
        output = F.log_softmax(x,dim=1)
        return output
        
net = Net()

import torch.optim as optim
# create a stochastic gradient descent optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# create a loss function
criterion = nn.CrossEntropyLoss()


# train network
num_epochs = 5000
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs.float(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    # print statistics
    if ((epoch+1) % 10 == 0):
        print("epoch: " + str(epoch + 1) + ", loss: " + str(running_loss/len(data_train)))
print('Finished Training')

# save trained model
torch.save(net.state_dict(), './digits_net.pth')


# test neural network on test data
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = net(inputs.float())
        _, predicted = torch.max(outputs.data.float(), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 300 test images: %d %%' % (
    100 * correct / total))

# print accuracy of each class separately
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = net(inputs.float())
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of Digit ' + str(classes[i]) + ': ' + str(100 * class_correct[i] / class_total[i]) + ' %')


# generate plot of results
from matplotlib.lines import Line2D
# plot decision boundary
x1 = np.linspace(-1.1,1.1,100)
x2 = np.linspace(-1.1,1.1,100)
test_plot = []
for i in range(100):
    for j in range(100):
        test_plot.append([x1[i],x2[j]])
test_plot = np.array(test_plot)
test_plot = MyDataset(test_plot)
testplotloader = DataLoader(test_plot, batch_size=300, shuffle=True)
with torch.no_grad():
    for data in testplotloader:
        inputs = data
        output = net(inputs.float())
        _, predicted = torch.max(output,1)
        inputs = inputs.numpy()
        for i in range(len(inputs)):
            p_color = colors[int(predicted[i])]
            plt.plot(inputs[i][0],inputs[i][1],color=p_color,marker='s',alpha=0.07,markeredgecolor='blue',markeredgewidth=0.0)

# plot points
plot_points(data_test, y_test)
colors = ['b','g','r','c','m','y','pink','orange','lime','deepskyblue']
lines = [Line2D([0], [0], marker='o', color=c, label='Scatter', markerfacecolor=c, markersize=5) for c in colors]

labels = ['0','1','2','3','4','5','6','7','8','9']
plt.legend(lines,labels,bbox_to_anchor=(1.05,1),loc='upper left')
plt.show()