import torch.nn as nn
import torch.nn.functional as F

class CNN1DNet(nn.Module):
    def __init__(self, input_dim=438, num_classes=12):
        super(CNN1DNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.flatten_dim = (input_dim // 4) * 64  # после двух pool(2)
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# TODO перенести экстарктор признаков в препроцессинг модели
class Preprocess:
    def __init__(self, step_file: str, rdf_k: int = 256, lbo_k: int = 16, bins: int = 64):
        self.step_file = step_file
        self.rdf_k = rdf_k
        self.lbo_k = lbo_k
        self.bins = bins

    def __call__(self, x):
        # Примените необходимые преобразования к входным данным
        return x