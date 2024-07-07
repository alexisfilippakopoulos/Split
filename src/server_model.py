from torch import nn, flatten

class ServerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=5*5*128, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=10)
        
    def forward(self, x):
        x = self.fc1(flatten(x, 1))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    server_model = ServerModel()

    print("ServerModel parameters:", count_parameters(server_model))

    