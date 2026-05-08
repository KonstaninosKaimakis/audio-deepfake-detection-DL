import torch.nn as nn

class DeepfakeDetectorMLP(nn.Module):
    def __init__(self, input_size=390):
        super(DeepfakeDetectorMLP, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1: Input to 512
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2: 512 to 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 3: 256 to 64
            nn.Linear(256, 64),
            nn.ReLU(),
            
            # Output Layer: 1 neuron for binary classification
            # We use 1 output and Sigmoid for Real vs Fake
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

# Initialize
model = DeepfakeDetectorMLP(input_size=390)