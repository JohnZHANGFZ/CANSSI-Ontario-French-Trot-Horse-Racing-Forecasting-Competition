import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import time
from torch.utils.data import Dataset
import torch


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







class HorseRacingDataset(Dataset):
    def __init__(self, csv_file) :
        self.data_frame = pd.read_csv(csv_file)

        # Group by RaceID and sort
        self.race_groups = self.data_frame.groupby('RaceID')

        self.feature_columns = ['Barrier', 'Distance', 'IntGender',
                                'HandicapDistance',
                                'HindShoes', 'HorseAge', 'IntRacingSubType',
                                'IntStartType', 'StartingLine', 'IntSurface']
        self.label_column = 'FinishPosition'

        # Preprocess each group separately
        self.races = [(race_id, self._preprocess_group(group)) for
                      race_id, group in self.race_groups]

    def _preprocess_group(self, group) :
        # Perform preprocessing on the group
        # For example, encode categorical variables and normalize
        # Here you would return the processed features and label
        return group

    def __len__(self) :
        return len(self.races)

    def __getitem__(self, idx) :
        race_id, race_data = self.races[idx]

        features = race_data[self.feature_columns]
        labels = race_data[self.label_column]

        # Convert features and labels to tensors
        features_tensor = torch.tensor(features.values.astype(np.float32))
        labels_tensor = torch.tensor(labels.values).long()

        return features_tensor, labels_tensor


def collate_fn(batch) :
    # Sort the batch in the descending order of number of horses
    batch.sort(key=lambda x : x[0].shape[0], reverse=True)

    # Separate features and labels, and pad features
    features = pad_sequence([item[0] for item in batch], batch_first=True,
                            padding_value=0)
    labels = pad_sequence([item[1] for item in batch], batch_first=True,
                          padding_value=-1)  # Assuming '-1' is an appropriate padding value for labels

    return features, labels
