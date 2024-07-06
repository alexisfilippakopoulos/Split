from torch.utils.data import Subset
from torch import save
from torchvision import datasets, transforms
import os


def get_data(transform):
    train_data = datasets.CIFAR10(root='data/', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='data/', train=False, download=True, transform=transform)
    return train_data, test_data

def sort_dataset(data):
    # Get unique classes
    unique_classes = list(data.class_to_idx.values())
    # Dict that gets the data indices for each unique label
    class_to_indices = {}
    for cl in unique_classes:
        class_to_indices[cl] = []
    
    for i in range(len(data)):
        class_to_indices[data.targets[i]].append(i)

    subsets = {}
    for cl, idxs in class_to_indices.items():
        subsets[cl] = Subset(dataset=data, indices=idxs)

    return subsets

def save_subsets(subsets: dict, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for cl, sub in subsets.items():
        print('Storing subset for class', cl)
        save(obj=sub, f=f'{folder}/subset_{cl}.pth')


if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data, test_data = get_data(transform=transform)
    subsets = sort_dataset(data=train_data)
    save_subsets(subsets=subsets, folder='subset_data')