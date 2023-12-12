# -*- coding: utf-8 -*-
# %%
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
# %%

BATCH_SIZE = 5000

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
        print(output4)
        self.fc1 = nn.Linear(49997, hidden_size2)
        print(f"int(output4)*hidden_size1 = {int(output4)*hidden_size1}")
        #self.fc1 = nn.Linear(int(output4)*hidden_size1, hidden_size2)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size2, output_size)
        #self.softmax=nn.Softmax(dim= 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(x.shape)
        x = x.view(1,-1)#1,980
        print(f"x.view(1,-1) {x.shape}")
        x = self.conv1(x) #12,48
        print(f"self.conv1(x) {x.shape}")
        x = self.relu1(x)
        print("relu done")
        x = self.pool1(x)#12,46
        print(f"self.pool1(x) {x.shape}")
        x = self.conv2(x)#1,47
        print(f"self.conv2(x) {x.shape}")
        x = self.relu2(x)
        print("relu2 done")
        x = self.pool2(x)#1,46
        print(f"self.pool2(x) {x.shape}")
        x = x.view(x.size(0),-1)
        print(f"x.view(x. {x.shape}")
        x = self.fc1(x)#1,16
        print(f"self.fc1(x) {x.shape}")
        x = self.relu3(x)
        print("relu3 done")
        x = self.fc2(x)
        print(f"self.fc2(x) {x.shape}")
        #print(x.shape)
        #x = self.softmax(x)
        x = self.sigmoid(x)
        return x.flatten()
# %%

def train_loop(X_train, epochs, learning_rate, model, l, y_train):
    batch_size = BATCH_SIZE
    pred_train=[]
    Nb_tweets=X_train.shape[0]
    y_train = y_train[:,0]
    # y_train = y_train.type(torch.LongTensor)
    y_train[y_train == -1] = 1  # between 0 and 1 instead of -1 and 0
    o=torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.train()
    # initialize an empty dictionary
    res = {'epoch': [], 'train_loss': []}
    for e in range(epochs):
        print(f"Epoch {e+1}\n-------------------------------")
        running_loss=0
        permutation = torch.randperm(X_train.size()[0])
        for i in range(0, X_train.size()[0], batch_size):
            if i%1000 == 0:
                print(f"Tweet {i}")
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            outputs = model(batch_x)
            print("outputs size =",outputs.shape)
            print("y_train =",y_train.shape)
            pred_train.append(outputs)
            o.zero_grad() # setting gradient to zeros, bc I don't wanna accumulate the grads of all layers
            loss = l(outputs, batch_y)
            loss.backward() # backward propagation
            o.step() # update the gradient to new gradients
            running_loss += loss.item()
        running_loss=running_loss/batch_size
        res['epoch'].append(e) # populate the dictionary of results
        res['train_loss'].append(running_loss)
    print("Done!")
    res = pd.DataFrame.from_dict(res) # translate the dictionary into a pandas dataframe
    res.to_csv("./metrics_over_epochs_0{:.5f}.csv".format(learning_rate), mode = 'w', index = False) # store the results into a *.csv file
    return res['train_loss'], pred_train

def test_loop(X_test,y_test, l, model):
    pred_test=[]
    nb_test=X_test.shape[0]
    test_loss=0
    model.eval()
    with torch.no_grad():
        # loop over all test input-output pairs
        for i in range(nb_test):
            outputs = model(X_test[i])
            pred_test.append(outputs)
            loss = l(outputs, y_test[i].unsqueeze(0).float()) # compute the loss
            test_loss += loss.item()
        test_loss = test_loss/nb_test
    return test_loss, pred_test

def test_loop_final(X_test, model):
    pred_test=[]
    nb_test=X_test.shape[0]
    test_loss=0
    model.eval()
    with torch.no_grad():
        # loop over all test input-output pairs
        for i in range(nb_test):
            outputs = model(X_test[i])
            pred_test.append(outputs)
    return pred_test
# %%

device = torch.device("cpu")
PREFIX = "twitter-datasets/"
X_train = torch.cat((torch.load(PREFIX+"X_train_cnn_new_0_full.pt", device),
            #torch.load(PREFIX+"X_train_cnn_new_1_full.pt", device),
            #torch.load(PREFIX+"X_train_cnn_new_2_full.pt", device),
            #torch.load(PREFIX+"X_train_cnn_new_3_full.pt", device),
            #torch.load(PREFIX+"X_train_cnn_new_4_full.pt", device),
            #torch.load(PREFIX+"X_train_cnn_new_5_full.pt", device)
            ), 0)

X_test = torch.cat((torch.load(PREFIX+"X_test_cnn_new_0_full.pt", device),
            #torch.load(PREFIX+"X_test_cnn_new_1_full.pt", device),
            #torch.load(PREFIX+"X_test_cnn_new_2_full.pt", device),
            #torch.load(PREFIX+"X_test_cnn_new_3_full.pt", device),
            #torch.load(PREFIX+"X_test_cnn_new_4_full.pt", device),
            #torch.load(PREFIX+"X_test_cnn_new_5_full.pt", device)
            ), 0)

y_train = torch.cat((torch.load(PREFIX+"y_train_cnn_new_0_full.pt", device),
            #torch.load(PREFIX+"y_train_cnn_new_1_full.pt", device),
            #torch.load(PREFIX+"y_train_cnn_new_2_full.pt", device),
            #torch.load(PREFIX+"y_train_cnn_new_3_full.pt", device),
            #torch.load(PREFIX+"y_train_cnn_new_4_full.pt", device),
            #torch.load(PREFIX+"y_train_cnn_new_5_full.pt", device)
            ), 0)

y_test = torch.cat((torch.load(PREFIX+"y_test_cnn_new_0_full.pt", device),
            #torch.load(PREFIX+"y_test_cnn_new_1_full.pt", device),
            #torch.load(PREFIX+"y_test_cnn_new_2_full.pt", device),
            #torch.load(PREFIX+"y_test_cnn_new_3_full.pt", device),
            #torch.load(PREFIX+"y_test_cnn_new_4_full.pt", device),
            #torch.load(PREFIX+"y_test_cnn_new_5_full.pt", device)
            ), 0)

#one hot encoded
#y_train_labels = torch.tensor([(0, 1) if label == -1 else (1, 0) for label in y_train[:,0]], device=device)
#y_test_labels = torch.tensor([(0, 1) if label == -1 else (1, 0) for label in y_test[:,0]], device=device)
X_T = torch.load(PREFIX+"X_T.pt", device)
# %%

# définir modele

max_tweet=X_train.shape[1]
nb_features=20
in_channels = 1
hidden_size = 12
hidden_size1 = BATCH_SIZE  #sinon ça collait pas dans train_loop pour le batch_size
hidden_size2 = 16
kernel_size = nb_features*2
output_size = 1


model = CNN(input_size=(1,max_tweet,nb_features),
        hidden_size=hidden_size,
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        kernel_size=kernel_size,
        output_size=output_size)

model = model.to(device)

epochs=2
learning_rate = np.logspace(-4,-3, 2)
l= torch.nn.CrossEntropyLoss()
# %%

plt.figure(figsize=(8, 6))
for lr in learning_rate:
    x = train_loop(X_train, epochs, lr, model, l, y_train)[0]
    plt.plot(np.arange(0,epochs,1),x, label=f'Learning rate = {lr}')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Train Error')
plt.grid(True)
plt.legend(loc='upper right', title='train loop', fontsize='medium')
plt.savefig('myplot.png')


# %%
test_loss, y_pred1= test_loop(X_test, y_test_labels, l, model)
print("test_loss", test_loss)
print("y_pred1", y_pred1)


y_pred_final =  test_loop_final(X_T, model)