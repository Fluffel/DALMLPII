import torch
import torch.nn as nn

class DiscriminativeMultilayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(DiscriminativeMultilayerPerceptron, self).__init__()
        
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        # self.linear4 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear5 = nn.Linear(in_features=hidden_size, out_features=2)

    def forward(self, x):
        # x = x.flatten()
        x = self.linear1(x)
        x = torch.relu(x)

        x = self.linear2(x)
        x = torch.relu(x)

        x = self.linear3(x)
        x = torch.relu(x)

        # x = self.linear4(x)
        # x = torch.relu(x)

        x = self.linear5(x)
        return x

class LeNetEmbeddingModel(nn.Module):
    def __init__(self, le_net_base):
        super(LeNetEmbeddingModel, self).__init__()
        self.conv1 = le_net_base.conv1
        self.conv2 = le_net_base.conv2
        
        self.linear1 = le_net_base.linear1
        self.linear2 = le_net_base.linear2
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2)
        
        x = x.flatten(start_dim=1, end_dim=-1)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)

        return x


class EarlyStop():
    def __init__(self, patience, epsilon):
        self.patience = patience
        self.epsilon = epsilon
        self.count = 0
        self.min_loss = float('inf')

    def early_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.count = 0
        elif loss > (self.min_loss + self.epsilon):
            self.count += 1
            if self.count > self.patience:
                print("STOPPED EARLY")
                return True
        return False

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # 1 * 28 * 28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))
        # 6 * 24 * 24 -> pooling() -> 6 * 12 * 12
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        # 16 * 8 * 8 -> pooling() -> 16 * 4 * 4
        self.linear1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=10)


    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2)

        x = x.flatten(start_dim=1, end_dim=-1)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        # x = torch.softmax(x, dim=1)
        return x
