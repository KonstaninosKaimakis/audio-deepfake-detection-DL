import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class AudioFeatureDataset(Dataset):
    def __init__(self, dataframe):
        # 1. Separate features and labels
        # We drop 'filename' because the NN can't process strings
        # We drop 'label' because it's the target
        self.features = dataframe.drop(columns=['label', 'filename']).values.astype('float32')
        self.labels = dataframe['label'].values.astype('int64')
        
        # 2. Convert to Tensors
        self.features = torch.tensor(self.features)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- Workflow ---

# 1. Load the data
df = pd.read_parquet('your_file.parquet')

# 2. Split into Train and Validation sets (80/20)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# 3. Scale the features (CRITICAL for Deep Learning)
# Neural networks struggle when features have wild ranges (e.g., -600 to 0.03)
scaler = StandardScaler()
feature_cols = df.drop(columns=['label', 'filename']).columns
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols] = scaler.transform(val_df[feature_cols])

# 4. Create Dataset objects
train_dataset = AudioFeatureDataset(train_df)
val_dataset = AudioFeatureDataset(val_df)

# 5. Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)