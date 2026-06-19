import torch.nn as nn


class CNNPoolingBaseline(nn.Module):
    def __init__(self, depth=3, base_channels=32, dropout=0.5, num_classes=1):
        super().__init__()
        blocks = []
        in_ch = 1
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            blocks += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(in_ch, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class RNNBaseline(nn.Module):
    def __init__(self, input_size=128, num_layers=2, hidden_size=128, dropout=0.5, num_classes=1):
        super().__init__()
        self.lstm_layers = nn.ModuleList()
        self.drops = nn.ModuleList()
        cur_input = input_size
        for i in range(num_layers):
            h = hidden_size // (2 ** i)
            self.lstm_layers.append(
                nn.LSTM(cur_input, h, batch_first=True, bidirectional=True)
            )
            self.drops.append(nn.Dropout(dropout))
            cur_input = h * 2
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(cur_input, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for lstm, drop in zip(self.lstm_layers, self.drops):
            x, _ = lstm(x)
            x = self.relu(x)
            x = drop(x)
        return self.classifier(x[:, -1, :])


class CRNNBaseline(nn.Module):
    def __init__(self, cnn_depth=3, base_channels=32, num_rnn_layers=2, rnn_hidden=128,
                 dropout=0.5, freq_bins=128, num_classes=1):
        super().__init__()
        blocks = []
        in_ch = 1
        for i in range(cnn_depth):
            out_ch = base_channels * (2 ** i)
            blocks += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)

        freq_after_pool = freq_bins // (2 ** cnn_depth)
        rnn_input_dim = in_ch * freq_after_pool

        self.lstm_layers = nn.ModuleList()
        self.drops = nn.ModuleList()
        cur_input = rnn_input_dim
        for i in range(num_rnn_layers):
            h = rnn_hidden // (2 ** i)
            self.lstm_layers.append(
                nn.LSTM(cur_input, h, batch_first=True, bidirectional=True)
            )
            self.drops.append(nn.Dropout(dropout))
            cur_input = h * 2
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(cur_input, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        batch_size, seq_len, channels, freq = x.size()
        x = x.view(batch_size, seq_len, channels * freq)
        for lstm, drop in zip(self.lstm_layers, self.drops):
            x, _ = lstm(x)
            x = self.relu(x)
            x = drop(x)
        return self.classifier(x[:, -1, :])
