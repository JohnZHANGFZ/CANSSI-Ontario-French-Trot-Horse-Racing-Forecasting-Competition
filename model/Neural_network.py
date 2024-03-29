import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

feature_columns = ['Barrier', 'Distance', 'IntGender', 'HandicapDistance',
                   'HindShoes', 'HorseAge', 'IntRacingSubType',
                   'IntStartType', 'StartingLine', 'IntSurface']
label_column = 'FinishPosition'


class HorseRacingDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)

        # Select the feature and label columns
        self.features = self.data_frame[
            ['Barrier', 'Distance', 'IntGender', 'HandicapDistance',
             'HindShoes', 'HorseAge', 'IntRacingSubType',
             'IntStartType', 'StartingLine', 'IntSurface']]
        self.labels = self.data_frame['FinishPosition']

        # Data preprocessing, such as coding categorical variables, standardization

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Extract individual sample features and labels
        sample = self.features.iloc[idx]
        label = self.labels.iloc[idx]

        # transfer to PyTorch Tensor
        sample_tensor = torch.tensor(sample.values.astype(np.float32))
        label_tensor = torch.tensor(label).long()  # assume FinishPosition is int

        return sample_tensor, label_tensor


data_path = '/Users/fangzhengzhang/Desktop/CANSSI/Training_2.csv'

train_set = HorseRacingDataset(data_path)

train_data = DataLoader(dataset=train_set, batch_size=64, shuffle=False)


# 1. define the model
class RegressionModel(nn.Module):
    def __init__(self, input_size):
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


# instantiate the model
input_size = len(feature_columns)  # feature nums
model = RegressionModel(input_size)

# 2. define loss function and optimizer
criterion = nn.MSELoss()  # Mean square error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam

# 3. prepare Dataloader
batch_size = 64
num_epochs = 30
dataset = HorseRacingDataset(csv_file='/Training_2.csv')
print("dataset complete loading")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print("dataloader complete loading")
loss_file = open("loss.csv", "w")
# 4. train the model
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(dataloader):
        # forward propagation
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels.float())  # calculate loss

        # backward propagation and optimization
        optimizer.zero_grad()  # clear past gradient
        loss.backward()  # backward propagation
        optimizer.step()  # updata parameter

        # print out loss information
        if (i + 1) % 100 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item()}')
            loss_file.write(str(loss.item()))
    model_file_path = f'/mnt/data/horse_racing_model_{epoch}.pth'
    torch.save(model.state_dict(), model_file_path)

# save model parameter
loss_file.close()
model_file_path = '/mnt/data/horse_racing_model_final.pth'
torch.save(model.state_dict(), model_file_path)
