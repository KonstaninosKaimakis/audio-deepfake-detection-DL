import torch
from torch.utils.data import Dataset

class AudioFeatureDataset(Dataset):
    def __init__(self, dataframe):
        # Drop metadata/labels to extract only features
        self.features = torch.tensor(
            dataframe.drop(columns=["label", "filename"]).values.astype("float32")
        )
        # float32 labels required by BCELoss
        self.labels = torch.tensor(
            dataframe["label"].values.astype("float32")
        ).unsqueeze(1)  

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]