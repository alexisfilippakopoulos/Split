from torch import nn, flatten

       
class WeakClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        return x
    
class WeakClientOffloadedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=5*5*128, out_features=10)
        
    def forward(self, x):
        x = self.fc1(flatten(x, 1))
        return x
    

class StrongClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(in_features=5*5*128, out_features=10)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x_off = self.pool(self.relu(self.conv4(x)))
        x = self.fc1(flatten(x_off, 1))
        return x, x_off
    

"""def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    weak_model = WeakClientModel()
    weak_off_model = WeakClientOffloadedModel()
    strong_model = StrongClientModel()

    print("WeakClientModel parameters:", count_parameters(weak_model))
    print("WeakClientOffloadedModel parameters:", count_parameters(weak_off_model))
    print("StrongClientModel parameters:", count_parameters(strong_model))"""
