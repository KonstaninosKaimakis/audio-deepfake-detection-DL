import torch.nn as nn


class DeepfakeDetectorMLP(nn.Module):
    def __init__(self, input_size=390, hidden_sizes=None, dropout=0.5):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]
        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers += [
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_features = hidden_size
        layers += [nn.Linear(in_features, 1), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
