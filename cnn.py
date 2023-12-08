import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import loops
import pandas as pd


class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size1, hidden_size2, kernel_size, output_size):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_size[0], out_channels=hidden_size, kernel_size=kernel_size, stride=20, padding=0)#(d - kernel_size + 1) / 1 = (10000 - 3 + 1) / 1 = 9999
        output1=(input_size[1]*input_size[2]-kernel_size+1)/20
        self.relu1 = nn.ReLU()

        self.pool_kernel_size = 3
        self.pool_stride = 1
        self.pool1 = nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        output2=(output1-self.pool_kernel_size+1)/1
        


        self.conv1d_kernel_size = 2
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size1, kernel_size=self.conv1d_kernel_size, stride=1, padding=1)
        output3=(output2-self.conv1d_kernel_size+2*1)/1+1
        self.relu2 = nn.ReLU()
        self.pool2_kernel_size = 2
        self.pool2 = nn.MaxPool1d(kernel_size=self.pool2_kernel_size, stride=self.pool_stride)
        output4=(output3-self.pool2_kernel_size+1)/self.pool_stride +1
        
        self.fc1 = nn.Linear(int(output4)*hidden_size1, hidden_size2)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size2, output_size)
        self.softmax=nn.Softmax(dim= 1)

    def forward(self, x):
        x = x.view(1,-1)#1,980
        x = self.conv1(x) #12,48
        x = self.relu1(x)
        x = self.pool1(x)#12,46
        x = self.conv2(x)#1,47
        x = self.relu2(x)
        x = self.pool2(x)#1,46
        x = x.view(x.size(0),-1)
        x = self.fc1(x)#1,16
        x = self.relu3(x)
        x = self.fc2(x)
        #print(x.shape)
        x = self.softmax(x)
        return x
    
#lire data avec padding
X_train = torch.load("X_train_cnn_new.pt")

X_test =  torch.load("X_test_cnn_new.pt")

y_train=torch.load("y_train_cnn_new.pt")
#one hot encoded
y_train_labels = torch.tensor([(0, 1) if label == -1 else (1, 0) for label in y_train[:,0]])

y_test=torch.load("y_test_cnn_new.pt")
y_test_labels = torch.tensor([(0, 1) if label == -1 else (1, 0) for label in y_test[:,0]])

X_T = torch.load("X_T.pt")



# définir modele

max_tweet=X_train.shape[1]
nb_features=20
in_channels = 1  
hidden_size = 12  
hidden_size1 = 1  #sinon ça collait pas dans train_loop pour le batch_size
hidden_size2 = 16  
kernel_size = nb_features*2
output_size = 2


model = CNN(input_size=(1,max_tweet,nb_features), 
        hidden_size=hidden_size,
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2, 
        kernel_size=kernel_size, 
        output_size=output_size)


epochs=2
learning_rate = np.logspace(-4,-3, 2)
l= torch.nn.CrossEntropyLoss()

plt.figure(figsize=(8, 6))
for lr in learning_rate:
    x=loops.train_loop(X_train, epochs, lr, model, l, y_train_labels)[0]
    plt.plot(np.arange(0,epochs,1),x, label=f'Learning rate = {lr}')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Train Error')
plt.grid(True)
plt.legend(loc='upper right', title='train loop', fontsize='medium')
plt.savefig('myplot.png')


test_loss, y_pred1=loops.test_loop(X_test, y_test_labels, l, model)
print("test_loss", test_loss)
print("y_pred1", y_pred1)


y_pred_final = loops.test_loop_final(X_T, model)
