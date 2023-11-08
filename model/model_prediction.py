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
        self.data_frame['FinishPosition'] = 0

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


def bracket(mylist):
    result = []
    for i in range(len(mylist[0])):
        result.append(mylist[0][i][0])
    return result

def insert(L, i, indexlist):
    value = L[i]
    value_index = indexlist[i]
    while i>0 and L[i-1] > value:
        L[i] = L[i-1]
        indexlist[i] = indexlist[i-1]
        i = i - 1
    L[i] = value
    indexlist[i] = value_index


def insertion_sort(L, indexlist):
    for i in range(len(L)):
        insert(L, i, indexlist)
    #print(L)
    #print(indexlist)


def prob(index_list):
    cur_prob = 0.6
    prob_list = []
    for i in range(len(index_list)):
        if i < len(index_list) - 1:
            prob_list.append(cur_prob)
            cur_prob = (1-sum(prob_list))/2
        else:
            prob_list.append(1-sum(prob_list))
    return prob_list


def to_prob(rank):
    result = []
    for i in range(0, len(rank)):
        predicted = []
        index_list = []
        probability = []
        for j in range(0, len(rank[i])):
            predicted.append(rank[i][j])
            index_list.append(j+1)
        probability = prob(index_list)
        insertion_sort(predicted, probability)
        for item in probability:
            result.append(item)
    return result


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


train_set = HorseRacingDataset(
    '/Users/fangzhengzhang/Desktop/CANSSI/Holdout_cleaned.csv')
train_data = DataLoader(dataset=train_set, batch_size=64, shuffle=False)

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
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
data_frame = pd.DataFrame(pd.read_csv('/Users/fangzhengzhang/Desktop/CANSSI/Holdout_cleaned.csv'))
forecasts = []
# sample_features, _ = dataset[0]
# sample_features = torch.tensor([your_features], dtype=torch.float32)
for i, (features, labels) in enumerate(train_loader):
        # model prediction,use torch.unsqueeze to add a batch processing dimension.
        sample_features = torch.unsqueeze(features, 0)  # adding dimension
        print("data access succeed")
        with torch.no_grad():  # make sure doesn't calculate gradient in eval mode
            print("begin to reasoning")
            prediction = model(sample_features)
            forecasts.extend(bracket(prediction.tolist()))
prob_num = []
data_frame["Forecasts"] = forecasts
data_frame['Rank'] = data_frame.groupby('RaceID')['Forecasts'].rank(method='min', ascending=True)
column_list = data_frame['Rank'].tolist()
all_nested_ranks = [group["Rank"].tolist() for _, group in data_frame.groupby('RaceID')]
for i in range(len(all_nested_ranks)):
    prob_num.extend(to_prob([all_nested_ranks[i]]))
data_frame["Probability"] = prob_num
data_frame.to_csv("holdout_with_prediction.csv")

