
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd



class fCNN(nn.Module):  
    def __init__(self, input_size, hidden_size, output_size):
            super(fCNN, self).__init__()
            self.flatten= nn.Flatten()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.softmax=nn.Softmax()

        
    def forward(self, x):
        print("x.ini", x.shape)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        #return x.flatten()
        return x


def train_loop(X_train, epochs, learning_rate, model, l, y_train):
    pred_train=[]
    X_train_float=X_train.to(torch.float32)
    N=X_train.shape[0]
    o=torch.optim.Adam(model.parameters(), lr=learning_rate)
    # initialize an empty dictionary
    res = {'epoch': [], 'train_loss': []}
    for e in range(epochs):
        print(f"Epoch {e+1}\n-------------------------------")
        running_loss=0
        for i in range(N):

            outputs = model(X_train_float[i])  
            pred_train.append(outputs)
            o.zero_grad() # setting gradient to zeros, bc I don't wanna accumulate the grads of all layers
            loss = l(outputs, y_train) 
            loss.backward() # backward propagation        
            o.step() # update the gradient to new gradients
            running_loss += loss.item()
        running_loss=running_loss/N
        res['epoch'].append(e) # populate the dictionary of results
        res['train_loss'].append(running_loss)
    print("Done!")
    res = pd.DataFrame.from_dict(res) # translate the dictionary into a pandas dataframe
    res.to_csv("./metrics_over_epochs_0{:.5f}.csv".format(learning_rate), mode = 'w', index = False) # store the results into a *.csv file
    return res['train_loss'], pred_train

X_train=pd.read_csv("X_train.csv")
X_test=pd.read_csv("X_test.csv")
y_train=pd.read_csv("y_train.csv")
y_train['label'] = y_train['label'].map({-1: (0, 1), 1: (1, 0)})

y_test=pd.read_csv("y_test.csv")
y_test['label'] = y_test['label'].map({-1: (0, 1), 1: (1, 0)})

X_train_torch=torch.tensor(X_train.values)
X_test_torch=torch.tensor(X_test.values)

y_train_torch=torch.tensor(y_train['label'].values.tolist(), dtype=torch.float32)
y_test_torch=torch.tensor(y_test['label'].values.tolist(), dtype=torch.float32)


output_size=2
p=20
model=fCNN(X_train_torch.shape[0],p,output_size)
params_fc1=model.fc1.weight
params_fc2=model.fc2.weight
l= torch.nn.CrossEntropyLoss()
epochs=50
learning_rate = np.logspace(-4,-3, 4)

plt.figure(figsize=(8, 6))
for lr in learning_rate:
    x=train_loop(X_train_torch, epochs, lr, model, l, y_train_torch)[0]
    plt.plot(np.arange(0,epochs,1),x, label=f'Learning rate = {lr}')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Train Error')
plt.grid(True)
plt.legend(loc='upper right', title='Légende', fontsize='medium')
plt.show()

def test_loop(X_test,y_test, l, model):
    pred_test=[]
    nb_test=X_test.shape[0]
    test_loss=0
    with torch.no_grad():
        # loop over all test input-output pairs
        for _ in range(nb_test): 
            outputs = model(X_test)  
            pred_test.append(outputs)
            loss = l(outputs, y_test) # compute the loss
            test_loss += loss.item()
        test_loss = test_loss/nb_test
    return test_loss, pred_test

y_pred=test_loop(X_test, y_test, l, model)[1]
