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

        # 选择特征列和标签列
        self.features = self.data_frame[
            ['Barrier', 'Distance', 'IntGender', 'HandicapDistance',
             'HindShoes', 'HorseAge', 'IntRacingSubType',
             'IntStartType', 'StartingLine', 'IntSurface']]
        self.labels = self.data_frame['FinishPosition']

        # 数据预处理，例如：编码分类变量、标准化等

    def __len__(self):
        # 返回数据集的大小
        return len(self.data_frame)

    def __getitem__(self, idx) :
        # 提取单个样本的特征和标签
        sample = self.features.iloc[idx]
        label = self.labels.iloc[idx]

        # 转换为PyTorch Tensor
        sample_tensor = torch.tensor(sample.values.astype(np.float32))
        label_tensor = torch.tensor(label).long()  # 假设FinishPosition是整数

        return sample_tensor, label_tensor


train_set = HorseRacingDataset(
    '/Users/fangzhengzhang/Desktop/CANSSI/Training_2.csv')

# 设置迭代器
train_data = DataLoader(dataset=train_set, batch_size=64, shuffle=False)



# 1. 定义模型
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


# 实例化模型
input_size = len(feature_columns)  # 特征数量
model = RegressionModel(input_size)

# 2. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 3. 准备数据加载器
batch_size = 64
num_epochs = 30
dataset = HorseRacingDataset(csv_file='/Users/fangzhengzhang/Desktop/CANSSI/Training_2.csv')
print("dataset 加载完毕")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print("dataloader 加载完毕")
loss_file = open("loss.csv", "w")
# 4. 训练模型
for epoch in range(num_epochs) :
    for i, (features, labels) in enumerate(dataloader) :
        # 前向传播
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels.float())  # 计算损失

        # 反向传播和优化
        optimizer.zero_grad()  # 清空过往梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 打印损失信息
        if (i + 1) % 100 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item()}')
            loss_file.write(str(loss.item()))
    model_file_path = f'/Users/fangzhengzhang/Desktop/CANSSI/model/mnt/data/horse_racing_model_{epoch}.pth'
    torch.save(model.state_dict(), model_file_path)

# 保存模型参数
loss_file.close()
model_file_path = '/Users/fangzhengzhang/Desktop/CANSSI/model/mnt/data/horse_racing_model_final.pth'
torch.save(model.state_dict(), model_file_path)
