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
