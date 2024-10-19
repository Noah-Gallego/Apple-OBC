import os
import pandas as pd
import torch
from torch.utils.data import Dataset

## a custom Dataset class that reads in a specifically-formatted pkl file activity data ##

# this class converts a pkl file into Tensors that can be batch processed 
class RunWalkDataset(Dataset):
    """Run or walk activity dataset class."""

    def __init__(self, pkl_file, transform=None):
        """
        :param pkl_file (string): Path to a binary pickle file.
        :param transform (callable, optional): Optional transform to be applied on a sample.
        """
        df = pd.read_pickle(pkl_file)

        # assumes all columns are input features except the last, which is the target
        input_features = df.iloc[:, :-1].values.astype(dtype = 'float32')
        target = df.iloc[:, -1:].values.astype(dtype = 'float32')
        
        # converts input features, target into tensors: x, y
        self.x = torch.tensor(input_features, dtype=torch.float32)
        self.y = torch.tensor(target, dtype=torch.float32)
        
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.transform:
            self.x = self.transform(self.x)
            self.y = self.transform(self.y)

        # returns input feat, target
        return self.x[idx], self.y[idx]

    
## training scripts that trains a model ##

# train loop
def train(model, train_loader, epochs, optimizer, criterion):
    """
    Training loop which returns a trained model (and saves it).
    
    :param model: the PyTorch model that we wish to train.
    :param train_loader: the DataLoader used for training.
    :param epochs: Total number of times to iterate through the training data.
    :param optimizer: optimizer to use during training.
    :param criterion: loss function used for training. 
    :return: trained model.
    """
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for data, target in train_loader:
            
            optimizer.zero_grad() # zero accumulated gradients
            
            # get output of model
            output = model(data)
            
            # calculate loss and perform backprop
            loss = criterion(output, target)
            
            # backprop + update step
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
        
        # print loss stats
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))
     
    # return trained model
    return model


## test script that evaluates the class accuracy of a model on a test set##

def test_eval(model, test_loader, criterion):

    # initialize test loss
    num_correct = 0.0

    model.eval() # prep model for evaluation 

    for data, target in test_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # round predictions to nearest class val (0, 1)
        predictions = torch.round(output)
        # see how many preds and targets match
        num_correct += torch.sum((predictions.data == target.data).float())
    
    return num_correct
