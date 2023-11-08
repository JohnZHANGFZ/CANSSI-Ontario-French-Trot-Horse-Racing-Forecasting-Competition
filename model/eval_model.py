import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import time
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


feature_columns = ['Barrier', 'Distance', 'IntGender', 'HandicapDistance',
                   'HindShoes', 'HorseAge', 'IntRacingSubType',
                   'IntStartType', 'StartingLine', 'IntSurface']
label_column = 'FinishPosition'


class HorseRacingDataset(Dataset):
    def __init__(self, csv_file):

        self.data_frame = pd.read_csv(csv_file)

        # select feature columns and label columns
        self.features = self.data_frame[
            ['Barrier', 'Distance', 'IntGender', 'HandicapDistance',
             'HindShoes', 'HorseAge', 'IntRacingSubType',
             'IntStartType', 'StartingLine', 'IntSurface']]
        self.labels = self.data_frame['FinishPosition']


    def __len__(self):
        # return the length of dataset
        return len(self.data_frame)

    def __getitem__(self, idx):
        # get a single sample's feature and label
        sample = self.features.iloc[idx]
        label = self.labels.iloc[idx]

        # transfer to PyTorch Tensor
        sample_tensor = torch.tensor(sample.values.astype(np.float32))
        label_tensor = torch.tensor(label).long()

        return sample_tensor, label_tensor


train_set = HorseRacingDataset(
    '/Users/fangzhengzhang/Desktop/CANSSI/Training_2.csv')

# set iterator
train_data = DataLoader(dataset=train_set, batch_size=64, shuffle=False)


class RegressionModel(nn.Module) :
    def __init__(self, input_size) :
        super(RegressionModel, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, 600)  #
        self.layer2 = torch.nn.Linear(600, 1200)  #
        self.layer3 = torch.nn.Linear(1200, 2400)
        self.layer4 = torch.nn.Linear(2400, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.relu(x)
        x = self.layer4(x)
        return x


# Instantiating the model
input_size = len(feature_columns)
model = RegressionModel(input_size)

# Loading the model parameters
model_path = '/Users/fangzhengzhang/Desktop/CANSSI/model/mnt/data/horse_racing_model.pth'
model.load_state_dict(torch.load(model_path))
# Setting the evaluation mode
model.eval()
print("model access succeed")

batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


# sample_features, _ = dataset[0]
# sample_features = torch.tensor([your_features], dtype=torch.float32)
for i, (features, labels) in enumerate(train_loader):
    if i < 20:
        # model prediction,use torch.unsqueeze to add a batch processing dimension.
        sample_features = torch.unsqueeze(features, 0)  # adding dimension
        print("data access succeed")
        with torch.no_grad():  # make sure doesn't calculate gradient in eval mode
            print("begin to reasoning")
            prediction = model(sample_features)
            print(labels.tolist())
            print(prediction.tolist())
    else:
        break


