
import torch 
import pandas as pd


def train_loop(X_train, epochs, learning_rate, model, l, y_train):
    pred_train=[]
    N=X_train.shape[0]
    o=torch.optim.Adam(model.parameters(), lr=learning_rate)
    # initialize an empty dictionary
    res = {'epoch': [], 'train_loss': []}
    for e in range(epochs):
        print(f"Epoch {e+1}\n-------------------------------")
        running_loss=0
        for i in range(N):

            outputs = model(X_train[i])  
            pred_train.append(outputs)
            o.zero_grad() # setting gradient to zeros, bc I don't wanna accumulate the grads of all layers
            loss = l(outputs, y_train[i]) 
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