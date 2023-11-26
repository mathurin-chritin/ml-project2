import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import fc
import pandas as pd


class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size1, hidden_size2, kernel_size, output_size):
        super(CNN, self).__init__()
        self.pool_kernel_size = input_size*2
        self.pool_stride = 20
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=kernel_size, stride=1, padding=0)#(d - kernel_size + 1) / 1 = (10000 - 3 + 1) / 1 = 9999
        output1=(input_size-kernel_size+1)/1
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        output2=(output1-kernel_size+1)/2
        self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size1, kernel_size=kernel_size, stride=1, padding=1)
        output3=(output2-kernel_size+2*1)/1+1
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        output4=(output3-self.pool_kernel_size+1)/self.pool_stride
        self.fc1 = nn.Linear(int(output4)**2, hidden_size2)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
    
#lire data avec padding

X_train=pd.read_csv("X_train_cnn.csv")
X_test=pd.read_csv("X_test_cnn.csv")

y_train=pd.read_csv("y_train_cnn.csv")
y_train['label'] = y_train['label'].map({-1: (0, 1), 1: (1, 0)})

y_test=pd.read_csv("y_test_cnn.csv")
y_test['label'] = y_test['label'].map({-1: (0, 1), 1: (1, 0)})

X_train_torch=torch.tensor(X_train.values)
X_test_torch=torch.tensor(X_test.values)
X_train_float=X_train_torch.to(torch.float32)
X_test_float=X_test_torch.to(torch.float32)

y_train_torch=torch.tensor(y_train['label'].values.tolist(), dtype=torch.float32)
y_test_torch=torch.tensor(y_test['label'].values.tolist(), dtype=torch.float32)



# définir modele

max_tweet=X_train_float.shape[1]
nb_features=20
in_channels = 1  
hidden_size = 12  
hidden_size1 = 1  #sinon ça collait pas dans train_loop pour le batch_size
hidden_size2 = 16  
kernel_size = nb_features*2
output_size = 2 


model = CNN(input_size=max_tweet, 
        hidden_size=hidden_size,
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2, 
        kernel_size=kernel_size, 
        output_size=output_size)


epochs=1
learning_rate = np.logspace(-4,-3, 4)
l= torch.nn.CrossEntropyLoss()

plt.figure(figsize=(8, 6))
for lr in learning_rate:
    x=fc.train_loop(X_train_float, epochs, lr, model, l, y_train_torch)[0]
    plt.plot(np.arange(0,epochs,1),x, label=f'Learning rate = {lr}')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Train Error')
plt.grid(True)
plt.legend(loc='upper right', title='train loop', fontsize='medium')
plt.show()

y_pred1=fc.test_loop(X_test, y_test, l, model)[1]
