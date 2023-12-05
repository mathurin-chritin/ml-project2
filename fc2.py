import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
#mypath = '/content/drive/MyDrive/'#modify your path
mypath = 'C:\epfl\ma1'
class fCNN(nn.Module):  
    def __init__(self, input_size, hidden_size, output_size):
        super(fCNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def train_loop(X_train, epochs, learning_rate, model, loss_fn, y_train):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    res = {'epoch': [], 'train_loss': []}

    for e in range(epochs):
        print(f"Epoch {e+1}\n-------------------------------")
        running_loss = 0

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss /= len(X_train)
        res['epoch'].append(e)
        res['train_loss'].append(running_loss)

    print("Done!")
    res_df = pd.DataFrame.from_dict(res)
    
    res_df.to_csv(f'{mypath}\metrics_over_epochs_0{learning_rate:.5f}.csv', mode='w', index=False)

    #res_df.to_csv(f'/content/drive/MyDrive/metrics_over_epochs_0{learning_rate:.5f}.csv', mode='w', index=False)
    return res_df['train_loss']

def test_loop(X_test, y_test, loss_fn, model):
    nb_test = len(X_test)
    test_loss = 0

    with torch.no_grad():
        outputs = model(X_test)
        loss = loss_fn(outputs, y_test)
        test_loss = loss.item()

    test_loss /= nb_test
    return test_loss, outputs

# Your data loading and preparation code
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_train['label'] = y_train['label'].map({-1: (0, 1), 1: (1, 0)})
y_test = pd.read_csv("y_test.csv")
y_test['label'] = y_test['label'].map({-1: (0, 1), 1: (1, 0)})

X_train_torch = torch.tensor(X_train.values)
X_test_torch = torch.tensor(X_test.values)
X_train_float = X_train_torch.to(torch.float32)
X_test_float = X_test_torch.to(torch.float32)

y_train_torch = torch.tensor(y_train['label'].values.tolist(), dtype=torch.float32)
y_test_torch = torch.tensor(y_test['label'].values.tolist(), dtype=torch.float32)

output_size = 2
p = 16
model = fCNN(X_train_torch.shape[1], p, output_size)
params_fc1 = model.fc1.weight
params_fc2 = model.fc2.weight
loss_fn = torch.nn.CrossEntropyLoss()

# Set batch size
batch_size = 32

# Training loop
epochs = 5
learning_rate = np.logspace(-4, -3, 4)

plt.figure(figsize=(8, 6))
for lr in learning_rate:
    train_losses = train_loop(X_train_float, epochs, lr, model, loss_fn, y_train_torch)
    plt.plot(np.arange(0, epochs, 1), train_losses, label=f'Learning rate = {lr}')

plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Train Error')
plt.grid(True)
plt.legend(loc='upper right', title='Legend', fontsize='medium')
plt.show()

# Testing loop
test_loss, y_pred = test_loop(X_test_float, y_test_torch, loss_fn, model)
print(f"Test Loss: {test_loss}")
